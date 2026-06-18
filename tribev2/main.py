# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Defines the main classes used in the experiment.

We suggest the following structure:
- `Data`: configures dataset and extractors to return DataLoaders
- `Trainer`: creates the deep learning model and exposes a `fit` and `test` methods
- `TribeExperiment`: main class that defines the experiment to run by using `Data` and `Trainer`
"""

import gc
import logging
import os
import typing as tp
from pathlib import Path

import neuralset as ns
import numpy as np
import pandas as pd
import pydantic
import torch
import yaml
from exca import ConfDict, TaskInfra
from neuralset.events.etypes import EventTypesHelper
from neuralset.events.utils import standardize_events
from neuraltrain.losses import BaseLoss
from neuraltrain.metrics import BaseMetric
from neuraltrain.models import BaseModelConfig
from neuraltrain.models.common import SubjectLayers
from neuraltrain.optimizers.base import BaseOptimizer
from neuraltrain.utils import BaseExperiment, WandbLoggerConfig
from torch import nn
from torch.utils.data import DataLoader

from .eventstransforms import *  # register custom events transforms in neuralset
from .model import *  # register custom models in neuraltrain
from .studies import *  # register studies
from .utils import (
    MultiStudyLoader,
    set_study_in_average_subject_mode,
    split_segments_by_time,
)
from .utils_fmri import *  # register TribeSurfaceProjector

# Configure logger
LOGGER = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", "%H:%M:%S")
_handler.setFormatter(_formatter)
if not LOGGER.handlers:
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)


def _free_extractor_model(extractor: ns.extractors.BaseExtractor) -> None:
    """
    * Delete cached GPU model from an extractor after its features are cached.
    * Extractors lazily load models onto GPU during ``prepare`` and keep them
    in ``_model``.  Since results are persisted to disk, the model is no
    longer needed afterwards and this frees VRAM for subsequent extractors.
    """
    targets = [extractor]
    if hasattr(extractor, "image"):
        targets.append(extractor.image)
    for target in targets:
        for attr in ("_model",):
            obj = getattr(target, attr, None)
            if isinstance(obj, torch.nn.Module):
                try:
                    delattr(target, attr)
                except Exception:
                    pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Data(pydantic.BaseModel):
    """Handles configuration and creation of DataLoaders from dataset and extractors."""

    model_config = pydantic.ConfigDict(extra="forbid") # raises an error if any extra fields are passed that are not defined in the class.

    study: MultiStudyLoader # !!!!!!!!!!!!!!!!!!!!!!!1
    # features
    neuro: ns.extractors.BaseExtractor
    text_feature: ns.extractors.BaseExtractor | None = None
    image_feature: ns.extractors.BaseExtractor | None = None
    audio_feature: ns.extractors.BaseExtractor | None = None
    video_feature: ns.extractors.BaseExtractor | None = None
    subject_id: ns.extractors.LabelEncoder = ns.extractors.LabelEncoder(
        event_field="subject", allow_missing=True, aggregation="first"
    ) # encodes subject indentifiers as integer labels for subject-specific layers
    frequency: float | None = None # if set: overrides the sampling frequency of all feature extractors
    features_to_use: list[
        tp.Literal["text", "audio", "video", "image", "context", "flow", "music"]  # which modalities to include in this experiment
    ]
    features_to_mask: list[
        tp.Literal["text", "audio", "video", "image", "context", "flow", "music"]  # modalities to zero out (masked but still passed through the pipeline)
    ] = []
    n_layers_to_use: int | None = None    # if set: selects this many evenly spaced layers from the extractor
    layers_to_use: list[float] | None = None  # if set: selects specific layers by fractional position (0=first, 1=last)
    layer_aggregation: tp.Literal["group_mean", "mean"] | None = "group_mean" # how to aggregate extractor layers before projection
    # --- Dataloader settings ---
    duration_trs: int = 40 # length of each data segment in TRs (fMRI repetition times)
    overlap_trs_train: int = 0  # number of overlapping TRs between consecutive training segments
    overlap_trs_val: int | None = None # number of overlapping TRs for validation segments (defaults to train value if None)
    batch_size: int = 64 # number of segments per batch
    num_workers: int | None = None # number of parallel workers for data loading
    shuffle_train: bool = True # whether to shuffle training segments each epoch
    shuffle_val: bool = False # whether to shuffle validation segments
    stride_drop_incomplete: bool = False  # if True: drop segments that don't fill the full duration
    split_segments_by_time: bool = False # if True: split train/val by time position rather than event labels

    def model_post_init(self, __context):
        """
        * Pydantic post-init hook: propagates layer selection and frequency settings to all active extractors
        """
        super().model_post_init(__context)
        # resolve which layers to use: either evenly spaced (n_layers_to_use) or explicit (layers_to_use)
        layers_to_use = None
        if self.n_layers_to_use is not None or self.layers_to_use is not None:
            assert not (
                self.n_layers_to_use is not None and self.layers_to_use is not None
            ), "Only one of n_layers_to_use or layers_to_use can be specified"
            if self.n_layers_to_use is not None:
                layers_to_use = np.linspace(0, 1, self.n_layers_to_use).tolist()
            else:
                layers_to_use = self.layers_to_use
        # propagate layer selection and aggregation method to each active modality extractor
        for modality in self.features_to_use:
            extractor = getattr(self, f"{modality}_feature")
            if hasattr(extractor, "layers"):
                setattr(extractor, "layer_aggregation", self.layer_aggregation)
                if layers_to_use is not None:
                    setattr(extractor, "layers", layers_to_use)
            # handle nested extractors (e.g. video extractor wrapping an image extractor)
            if hasattr(extractor, "image") and hasattr(extractor.image, "layers"):
                setattr(extractor.image, "layer_aggregation", self.layer_aggregation)
                if layers_to_use is not None:
                    setattr(extractor.image, "layers", layers_to_use)
        # propagate frequency override to all active extractors if set
        if self.frequency is not None:
            for modality in self.features_to_use:
                extractor = getattr(self, f"{modality}_feature")
                if hasattr(extractor, "frequency"):
                    setattr(extractor, "frequency", self.frequency)

    @property
    def TR(self) -> float:# returns the fMRI repetition time in seconds (inverse of acquisition frequency)
        
        return 1 / self.neuro.frequency

    def get_events(self) -> pd.DataFrame:# loads and returns the events table (stimuli timings per timeline)
        events = self.study.run() # study.run() # *************************************************************************** D3 (study is of type : class  MultiStudyLoader) [ study.run() function is inherited from class father EventsStudyBuidler and the child class MultiStudyLoader overrides it ][ Note that class MultiStudyLoader is defined on tribev2/utils.py] 
        events = events[events.type != "Sentence"]# exclude sentence-level events (use word-level instead)
         # log a summary of events grouped by study, split and type for debugging
        cols = ["index", "subject", "timeline"]
        event_summary = (
            events.reset_index().groupby(["study", "split", "type"])[cols].nunique()
        )
        LOGGER.info("Event summary: \n%s", event_summary)
        return events

    def get_loaders( # builds and returns DataLoaders for train and/or val splits # this the first higher connection point with data ,triggred in TribeExperiment run()
         
        self,
        events: pd.DataFrame | None = None,
        split_to_build: tp.Literal["train", "val", "all"] | None = None,
    ) -> tuple[dict[str, DataLoader], int]:

        if events is None:
            events = self.get_events() # load events if not provided  # ************************************************* D2 
        else:
            events = standardize_events(events)  # normalize format if provided externally
        # collect active feature extractors by modality name
        extractors = {}
        for modality in self.features_to_use:
            extractors[modality] = getattr(self, f"{modality}_feature")
        if "Fmri" in events.type.unique():
            extractors["fmri"] = self.neuro # add fMRI extractor only if fMRI events exist
        """
        * create one dummy CategoricalEvent per timeline to anchor segment extraction
        * each dummy event spans the full timeline duration and carries split/subject info
        """
        dummy_events = []
        for timeline_name, timeline in events.groupby("timeline"): # ************************************************* D8'(post Chain/chunking)
            if "split" in timeline.columns:
                splits = timeline.split.dropna().unique()
                assert (
                    len(splits) == 1
                ), f"Timeline {timeline_name} has multiple splits: {splits}"
                split = splits[0]
            else:
                split = "all"
            dummy_event = {
                "type": "CategoricalEvent",
                "timeline": timeline_name,
                "start": timeline.start.min(),
                "duration": timeline.stop.max() - timeline.start.min(),
                "split": split,
                "subject": timeline.subject.unique()[0],
            }
            dummy_events.append(dummy_event)
        events = pd.concat([events, pd.DataFrame(dummy_events)]) # ************************************************* D8'
        events = standardize_events(events) # ************************************************* D8'

        extractors["subject_id"] = self.subject_id # add subject id encoder as an extractor
        # remove extractors whose event types are not present in the events table " events types are same as modalities"
        #-------Remove incompatible Extractors -----
        """
        * Check the type of events present in the target study --> remove extractors that don't handle any of these events types
        * Avoid preparing extractors on data they have no events for(else ,it may crash or produce empty/invalide outputs)
        """
        features_to_remove = set()
        for extractor_name, extractor in extractors.items():
            event_types = EventTypesHelper(extractor.event_types).names
            if not any(
                [event_type in events.type.unique() for event_type in event_types]
            ):
                features_to_remove.add(extractor_name)
        for extractor_name in features_to_remove: 
            del extractors[extractor_name]
            LOGGER.warning(
                "Removing extractor %s as there are no corresponding events",
                extractor_name,
            )
        # prepare each extractor: aligns features to event timings and caches results
        for name, extractor in extractors.items():
            """
            * The full events dataframe is passed to all extractors.
            * Each extractor internally filters for its own relevant event types.
            """
            LOGGER.info("Preparing extractor: %s", name)
            # ******************************************************************************* D9 [pass chuncked data resulted from Chain in MuliStudyLoader in utils.py]
            extractor.prepare(events) # launches feature extraction on data returned from  events = self.get_events() , after adding dummy events : line 208-events = pd.concat([events, pd.DataFrame(dummy_events)])
            _free_extractor_model(extractor)# free GPU memory after feature extraction

        # Prepare dataloaders
        
        # --- Build DataLoaders per split --- # train / val spliting ?
        loaders = {}]
        if split_to_build is None:
            splits = ["train", "val"] # build both by default
        else:
            splits = [split_to_build]
        for split in splits:
            LOGGER.info("Building dataloader for split %s", split)
            if split == "all" or self.split_segments_by_time:
                split_sel = [True] * len(events)  # use all events
                shuffle = False
                overlap_trs = self.overlap_trs_train
            else:
                split_sel = events.split == split  # filter the events belonging to this split
                if split not in events.split.unique():
                    shuffle = False
                else:
                    shuffle = (
                        self.shuffle_train if split == "train" else self.shuffle_val
                    )
                if split == "val":
                    overlap_trs = self.overlap_trs_val or self.overlap_trs_train
                else:
                    overlap_trs = self.overlap_trs_train

            sel = np.array(split_sel)
             # slice the selected events into fixed-duration overlapping segments
             # one sliding window of 100 TRs cut from a timeline [segmentaion right befor the trasnformer and after feature extraction]
            segments = ns.segments.list_segments(
                events[sel],
                triggers=events[sel].type == "CategoricalEvent",# segment boundaries defined by dummy events
                stride=(self.duration_trs - overlap_trs) * self.TR, # step size in seconds between segments
                duration=self.duration_trs * self.TR, # segment length in seconds
                stride_drop_incomplete=self.stride_drop_incomplete,
            )
            # alternative train/val split: divide segments by their time position instead of event labels
            if self.split_segments_by_time:
                LOGGER.info(f"Total number of segments: {len(segments)}")
                segments = split_segments_by_time(
                    segments,
                    val_ratio=self.study.transforms["split"].val_ratio,
                    split=split,
                )
                LOGGER.info(f"# {split} segments: {len(segments)}")
            if len(segments) == 0:
                LOGGER.warning("No events found for split %s", split)
                continue
            # build PyTorch dataset from segments and extractors
            dataset = ns.dataloader.SegmentDataset(
                extractors=extractors,
                segments=segments,
                remove_incomplete_segments=False,  # keep partial segments at boundaries
            )
            dataloader = dataset.build_dataloader(
                shuffle=shuffle,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
            )
            loaders[split] = dataloader

        return loaders


class TribeExperiment(BaseExperiment):
    """Defines the main experiment pipeline including data loading and training/evaluation."""

    model_config = pydantic.ConfigDict(extra="forbid")

    data: Data
    # Reproducibility
    seed: int | None = 33
    # Model
    brain_model_config: BaseModelConfig
    # Loss
    loss: BaseLoss
    # Optimization
    optim: BaseOptimizer
    # Metrics
    metrics: list[BaseMetric]
    monitor: str = "val/pearson"
    # Weights & Biases
    wandb_config: WandbLoggerConfig | None = None
    # Hardware
    accelerator: str = "gpu"
    # Optim
    n_epochs: int | None = 10
    max_steps: int = -1
    patience: int | None = None
    limit_train_batches: int | None = None
    accumulate_grad_batches: int = 1
    # Others
    enable_progress_bar: bool = True
    log_every_n_steps: int | None = None
    fast_dev_run: bool = False
    save_checkpoints: bool = True
    checkpoint_filename: str = "best"
    resize_subject_layer: bool = False
    freeze_backbone: bool = False
    # Eval
    average_subjects: bool = False
    checkpoint_path: str | None = None
    load_checkpoint: bool = True
    test_only: bool = False

    # Internal properties
    _trainer: tp.Any = None
    _model: tp.Any = None
    _logger: tp.Any = None

    # Others
    infra: TaskInfra = TaskInfra(version="1")

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if self.infra.folder is None:
            msg = "infra.folder needs to be specified to save the results."
            raise ValueError(msg)
        # Update Trainer parameters based on infra
        self.infra.tasks_per_node = self.infra.gpus_per_node
        self.infra.slurm_use_srun = True if self.infra.gpus_per_node > 1 else False
        if self.infra.gpus_per_node > 1:
            self.metrics = [m for m in self.metrics if m.name not in ["TopkAcc"]]
            self.data.batch_size = self.data.batch_size // self.infra.gpus_per_node
        if self.accumulate_grad_batches > 1:
            self.data.batch_size = self.data.batch_size // self.accumulate_grad_batches

        if (
            not (self.checkpoint_path and self.load_checkpoint)
        ) or self.resize_subject_layer:
            study_summary = self.data.study.study_summary()
            self.data.subject_id.predefined_mapping = {
                subject: i for i, subject in enumerate(study_summary.subject.unique())
            }
            self.brain_model_config.subject_layers.n_subjects = (
                study_summary.subject.nunique()
            )
            if isinstance(self.brain_model_config.projector, SubjectLayers):
                self.brain_model_config.projector.n_subjects = (
                    study_summary.subject.nunique()
                )

        if self.average_subjects:
            study_name = self.data.study.names
            self.brain_model_config.subject_layers.average_subjects = True
            self.brain_model_config.subject_layers.n_subjects = 0
            if isinstance(self.brain_model_config.projector, SubjectLayers):
                self.brain_model_config.projector.average_subjects = True
            self.data.neuro.aggregation = "mean"
            self.data.subject_id.predefined_mapping = None
            if isinstance(study_name, str):
                LOGGER.debug(f"Setting study {study_name} in average subject mode")
                trigger_type = (
                    "Video" if study_name in ["Wen2017", "Allen2022Bold"] else "Audio"
                )
                self.data.study = set_study_in_average_subject_mode(
                    self.data.study, trigger_type=trigger_type, trigger_field="filepath"
                )
            else:
                pass
                # LOGGER.warning(
                #     "Cannot set study in average subject mode with multiple studies"
                # )

    def _get_checkpoint_path(self) -> Path | None:
        if self.checkpoint_path:
            assert Path(
                self.checkpoint_path
            ).exists(), f"Checkpoint path {self.checkpoint_path} does not exist."
            checkpoint_path = Path(self.checkpoint_path)
        else:
            checkpoint_path = Path(self.infra.folder) / "last.ckpt"
            if not checkpoint_path.exists():
                checkpoint_path = None
        return checkpoint_path

    def _init_module(self, model: nn.Module) -> tp.Any:
        from .pl_module import BrainModule

        checkpoint_path = self._get_checkpoint_path()
        if (
            self.load_checkpoint
            and checkpoint_path is not None
            and not self.resize_subject_layer
        ):
            LOGGER.info(f"Loading model from {checkpoint_path}")
            init_fn = BrainModule.load_from_checkpoint
            init_kwargs = {"checkpoint_path": checkpoint_path, "strict": False}
        else:
            init_fn = BrainModule
            init_kwargs = {}

        metrics = {
            split + "/" + metric.log_name: metric.build()
            for metric in self.metrics
            for split in ["val", "test"]
        }
        metrics = nn.ModuleDict(metrics)
        pl_module = init_fn(
            model=model,
            loss=self.loss.build(),
            optim_config=self.optim,
            metrics=metrics,
            config=ConfDict(self.model_dump()),
            **init_kwargs,
        )

        if self.resize_subject_layer:
            LOGGER.info("Resizing subject layer")
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint["state_dict"]
            weights = state_dict["model.predictor.weights"]
            _, in_channels, out_channels = weights.shape
            n_subjects = self.brain_model_config.subject_layers.n_subjects
            if self.brain_model_config.subject_layers.subject_dropout:
                n_subjects += 1
            if "model.predictor.bias" in state_dict:
                bias = state_dict["model.predictor.bias"]
                new_bias = torch.nn.Parameter(torch.zeros(n_subjects, out_channels))
                new_bias.data[:] = bias.mean(dim=0).repeat(n_subjects, 1)
                state_dict["model.predictor.bias"] = new_bias
            if self.freeze_backbone:
                for param in pl_module.parameters():
                    param.requires_grad = False
            for param in pl_module.model.predictor.parameters():
                param.requires_grad = True
            if (
                self.brain_model_config.low_rank_head is not None
                and self.brain_model_config.low_rank_head != in_channels
            ):
                r = self.brain_model_config.low_rank_head
                if "model.low_rank_head.weight" in state_dict:
                    W1, W2 = (
                        state_dict["model.low_rank_head.weight"].cpu(),
                        state_dict["model.predictor.weights"].mean(dim=0).cpu(),
                    )
                    prod = torch.matmul(W1.t(), W2)
                else:
                    prod = state_dict["model.predictor.weights"].mean(dim=0).cpu()
                U, S, V = torch.svd(prod)
                U = U[:, :r]
                S = S[:r]
                V = V[:, :r]
                state_dict["model.low_rank_head.weight"] = U.t()
                state_dict["model.predictor.weights"] = torch.matmul(
                    torch.diag(S), V.t()
                ).repeat(n_subjects, 1, 1)
                if "model.predictor.bias" in state_dict:
                    state_dict["model.low_rank_head.bias"] = torch.zeros(r)
                for param in pl_module.model.low_rank_head.parameters():
                    param.requires_grad = True
            else:
                state_dict["model.predictor.weights"] = weights.mean(dim=0).repeat(
                    n_subjects, 1, 1
                )
            pl_module.load_state_dict(state_dict, strict=False)

        return pl_module

    def _setup_trainer(
        self, train_loader: DataLoader, override_n_devices: int | None = None
    ) -> tp.Any:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import (
            EarlyStopping,
            LearningRateMonitor,
            ModelCheckpoint,
        )

        batch = next(iter(train_loader))
        feature_dims = {}
        for modality in self.data.features_to_use:
            if (
                modality in batch.data and modality not in self.data.features_to_mask
            ):  # B, L, D, T
                if batch.data[modality].ndim == 4:
                    feature_dims[modality] = (
                        batch.data[modality].shape[1],
                        batch.data[modality].shape[2],
                    )
                elif batch.data[modality].ndim == 3:
                    feature_dims[modality] = (
                        1,
                        batch.data[modality].shape[1],
                    )
                else:
                    raise ValueError(
                        f"Unexpected number of dimensions for modality {modality}: {batch.data[modality].ndim}"
                    )
            else:
                feature_dims[modality] = None
        if "fmri" in batch.data:  # read from fmri config
            fmri = batch.data["fmri"]
            n_outputs = fmri.shape[1]
            for metric in self.metrics:
                if hasattr(metric, "kwargs") and "num_outputs" in metric.kwargs:
                    metric.kwargs["num_outputs"] = n_outputs
        else:  # read from neuro config
            if hasattr(self.data.neuro.projection, "mesh"):
                from neuralset.extractors.neuro import FSAVERAGE_SIZES

                n_outputs = 2 * FSAVERAGE_SIZES[self.data.neuro.projection.mesh]
            else:
                raise ValueError(
                    f"Could not determine number of outputs for neuro extractor {self.data.neuro}"
                )
        brain_model = self.brain_model_config.build(
            feature_dims=feature_dims,
            n_outputs=n_outputs,
            n_output_timesteps=self.data.duration_trs,
        )
        LOGGER.info("Extractor dims: %s", feature_dims)
        input_data = brain_model.aggregate_features(batch)
        LOGGER.info("Input shapes: %s", input_data.shape)
        LOGGER.info("Target shapes: %s", n_outputs)
        _ = brain_model(batch)
        total_params = sum(p.numel() for p in brain_model.parameters())
        LOGGER.info(f"Total parameters: {total_params}")
        self._model = self._init_module(brain_model)
        if self.monitor == "val/pearson":
            mode = "max"
        else:
            mode = "min"
        callbacks = [
            LearningRateMonitor(logging_interval="epoch"),
        ]
        if self.patience is not None:
            callbacks.append(
                EarlyStopping(monitor=self.monitor, mode=mode, patience=self.patience)
            )
        if self.save_checkpoints:
            callbacks.append(
                ModelCheckpoint(
                    save_last=True,
                    save_top_k=1,
                    dirpath=self.infra.folder,
                    filename=self.checkpoint_filename,
                    monitor=self.monitor,
                    mode=mode,
                    save_on_train_epoch_end=True,
                )
            )

        trainer = pl.Trainer(
            strategy="auto" if self.infra.gpus_per_node == 1 else "fsdp",
            devices=override_n_devices or self.infra.gpus_per_node,
            accelerator=self.accelerator,
            max_epochs=self.n_epochs,
            max_steps=self.max_steps,
            limit_train_batches=self.limit_train_batches,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=self.log_every_n_steps,
            fast_dev_run=self.fast_dev_run,
            callbacks=callbacks,
            logger=self._logger,
            enable_checkpointing=self.save_checkpoints,
            accumulate_grad_batches=self.accumulate_grad_batches,
        )
        self._trainer = trainer
        return trainer

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        self._trainer.fit(
            model=self._model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=self._get_checkpoint_path(),
        )

    def test(self, test_loader: DataLoader) -> None:
        if self.checkpoint_path:
            ckpt_path = self.checkpoint_path
        else:
            if self.save_checkpoints:
                ckpt_path = Path(self.infra.folder) / "best.ckpt"
            else:
                ckpt_path = None
        self._trainer.test(
            self._model,
            dataloaders=test_loader,
            ckpt_path=ckpt_path,
        )

    def setup_run(self):

        if self.infra.cluster and self.infra.status() != "not submitted":
            for out_type in ["stdout", "stderr"]:
                old_path = Path(getattr(self.infra.job().paths, out_type))
                new_path = Path(self.infra.folder) / f"log.{out_type}"
                try:
                    if new_path.exists():
                        os.remove(new_path)
                    os.symlink(
                        old_path,
                        new_path,
                    )
                except Exception:
                    pass
        config_path = Path(self.infra.folder) / "config.yaml"
        os.makedirs(self.infra.folder, exist_ok=True)
        with open(config_path, "w") as outfile:
            yaml.dump(
                self.model_dump(),
                outfile,
                indent=4,
                default_flow_style=False,
                sort_keys=False,
            )

    @infra.apply
    def run(self):
        import lightning.pytorch as pl

        self.setup_run()
        self._logger = (
            self.wandb_config.build(
                save_dir=self.infra.folder,
                xp_config=self.model_dump(),
                id=f"{self.wandb_config.group}-{self.infra.uid().split('-')[-1]}",
            )
            if self.wandb_config
            else None
        )

        if self.seed is not None:
            pl.seed_everything(self.seed, workers=True)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        loaders = self.data.get_loaders( # ***************************************************************** D1 ( data is of class Data )
            split_to_build="val" if self.test_only else None
        )
        self._setup_trainer(next(iter(loaders.values())))

        if not self.test_only:
            self.fit(loaders["train"], loaders["val"])

        self.test(loaders["val"])
