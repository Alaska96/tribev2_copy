# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree., 

"""Default configuration dictionary for TRIBE v2 experiments."""
import os
from pathlib import Path

PROJECT_NAME = "tribe_v2_baseline"

SLURM_PARTITION = os.getenv("SLURM_PARTITION", "only-one-gpu")
SLURM_CONSTRAINT = os.getenv("SLURM_CONSTRAINT", "")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "alaska0-university-of-milano-bicocca")
DATADIR = "/scratch_share/islab/Chaima/tribe_v1_work_space/Data/CMD_Data"
BASEDIR = "/scratch_share/islab/Chaima/tribe_v2_work_space"
CACHEDIR = os.path.join(BASEDIR, "cache", PROJECT_NAME)
SAVEDIR = os.path.join(BASEDIR, "results", PROJECT_NAME)
N_CPUS = 8  #20 # may need to be changed if it violate QOS policy

for path in [CACHEDIR, SAVEDIR, DATADIR]:
    Path(path).mkdir(parents=True, exist_ok=True)

text_feature = {
    "name": "HuggingFaceText",
    "event_types": "Word",
    "model_name": "meta-llama/Llama-3.2-3B",
    "aggregation": "sum",
    "frequency": 2, # One embedding per 0.5s
    "contextualized": True,
    "layers": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "batch_size": 4,
}
image_feature = {
    "name": "HuggingFaceVideo",
    "frequency": 2,  # One embedding per 0.5s #/ iit also represents the read frequency from the 60s chunk
    "event_types": "Video",
    "aggregation": "sum",
    "image": {
        "name": "HuggingFaceImage",
        "model_name": "facebook/dinov2-large",
        "layers": 2 / 3,
        "infra": {"keep_in_ram": False},
        "batch_size": 4,
    },
}
video_feature = image_feature | {
    "clip_duration": 4, 
    "image": {
        "name": "HuggingFaceImage",
        "model_name": "facebook/vjepa2-vitg-fpc64-256",# # facebook/vjepa2-vitl-fpc64-256
        "device": "cuda", # !!!!!!!!!! this what solved the video extractor bottelneck
        "infra": {"keep_in_ram": False},
        "layers": [0.75, 1.0],
    },
}
audio_feature = {
    "name": "Wav2VecBert",
    "frequency": 2,# One embedding per 0.5s
    "layers": [0.75, 1.0],
    "event_types": "Audio",
    "aggregation": "sum", 
}
neuro_extractor = {
    "name": "FmriExtractor",
    "allow_missing": True,
    "offset": 5,
    "frequency": 1/1.49, # was 1 but put it to 1/1.49 since i will use fMRI at TR=1.49s--> thus neuro_extractor will not apply fMRI resampling 
    "projection": {
        "name": "SurfaceProjector",# default projector ,works for run_cortical , but run_subcortical should override it to "MaskPorjector", and for my pipline i will disable the projector in further steps , since i will not use Algonaut2025Bold study
        "mesh": "fsaverage5",
        "kind": "ball",
        "radius": 3,
    },
}
for extractor in [
    text_feature,
    image_feature,
    video_feature,
    audio_feature,
    neuro_extractor,
]:
    extractor["infra"] = {
        "cluster": "slurm",
        "cpus_per_task": 8,# Number of CPUs per child job(per extractor)
        "mem_gb": 32, #64 to avoid small default memory amount assignement for child jobs which got them killed (to be double checked)
        #"slurm_setup": [f"export LD_LIBRARY_PATH={NVIDIA_LIBS}:$LD_LIBRARY_PATH"],# solves cudnn crash for audio extractor,# prepends tribe_v2_env's cuDNN 9.1 to library search path so it is loaded before the system cuDNN 9.0 (which lacks cudnnGetLibConfig)
        "folder": CACHEDIR,
        "keep_in_ram": False, # if True ,extracted features will be loaded to RAM after each extractor finishes, else they will be loaded during training 
        "mode": "cached",
        "min_samples_per_job": 100,   # min sample assigned to single job
        "max_jobs": 8,# was 256, hit QOS limit ,violating number of job submissions allowed atone go
        "timeout_min": 60 * 12*4, # 2 days 
        "slurm_partition": SLURM_PARTITION,
    }
    extractor["infra"]["version"] = "release"
    if extractor["name"] == "FmriExtractor":
        extractor["infra"]["max_jobs"] = 8#  was 1024 --> QOS limit
    else:
        extractor["infra"]["gpus_per_node"] = 1
        extractor["infra"]["slurm_constraint"] = SLURM_CONSTRAINT
        #extractor["device"] = "cuda" # !!!!!!!!!!!!!!!!!!! solve the issue of using CPU intead of GPU , alternative solution was to launch run from inside the compute node instead of log in node
    if extractor["name"] == "HuggingFaceVideo":
       # extractor["device"] = "cuda"
        extractor["infra"]["min_samples_per_job"] = 100 ## was 1
        extractor["infra"]["max_jobs"] = 8 #  was 1024 --> QOS limit
        extractor["infra"]["timeout_min"] = 60 * 24*2
    if extractor["name"] == "HuggingFaceText":
        extractor["infra"]["min_samples_per_job"] = 100   # was 32 
    extractor["allow_missing"] = True #f some chunks fail or are missing, don't crash, continue with what's available
    extractor["=replace="] = False # if set to True: tells exca to replace cached results if they exist rather than skipping — forces recomputation

default_config = {
    "infra": {
        "cluster": "slurm",
        "slurm_partition": SLURM_PARTITION,
        "folder": SAVEDIR,
        "gpus_per_node": 1,
        "cpus_per_task": N_CPUS,
        "mem_gb": 128, # was 64,set back to 128 in the train phase
        "timeout_min": 60 * 24*2,
        "mode": "retry",
        "slurm_constraint": SLURM_CONSTRAINT,
        "workdir": None,
    },
    "data": {
        "frequency": 2, # 2 embeddings per 1
        "duration_trs": 100, # segmentation window size in trs
        "overlap_trs_train": 0, 
        "overlap_trs_val": 0,
        "shuffle_val": True,
        "num_workers": N_CPUS,# 
        "layers_to_use": [0.5, 0.75, 1.0],#for the training phase
        "layer_aggregation": "group_mean",
        "study": {
            "names": [
                "Algonauts2025",# "Algonauts2025Bold" # algonauts_2025.competitors
               # "Wen2017",
               # "Lahner2024Bold",
               # "Lebel2023Bold",
            ],
            "path": DATADIR,
            "query": None,
            "infra_timelines": {#infra config specifically for the timeline/events generation phase (building the events DataFrame), not feature extraction

                "folder": CACHEDIR,
                "timeout_min": 60 * 12*4,
                "min_samples_per_job": 4,
                "max_jobs": 8, # "max_jobs": 1024, QOS limit
                "version": "final",
            },
            "transforms": {#pipeline of operations applied to events before extraction — order matters:
                "extractaudio": {"name": "ExtractAudioFromVideo"},#pulls audio track from video files
                "extractwords": {"name": "ExtractWordsFromAudio"},# extractwords: transcribes audio to words
                "addtext": {"name": "AddText"},#addtext/addsentence/addcontext: enriches word events with text context
                "addsentence": {
                    "name": "AddSentenceToWords",
                    "max_unmatched_ratio": 0.05,
                },
                "addcontext": {
                    "name": "AddContextToWords",
                    "sentence_only": False,
                    "max_context_len": 1024,
                    "split_field": "",
                },
                "removemissing": {"name": "RemoveMissing"},#drops events with missing data
                "chunksounds": { #splits long Audio events into 30-60s chunks
                    "name": "ChunkEvents",
                    "event_type_to_chunk": "Audio",
                    "max_duration": 60,
                    "min_duration": 30,
                },
                "chunkvideos": {# splits long ideo events into 30-60s chunks
                    "name": "ChunkEvents",
                    "event_type_to_chunk": "Video",
                    "max_duration": 60,
                    "min_duration": 30,
                    "infra": {"backend": "Cached", "folder": CACHEDIR},
                },
                "query": {"name": "QueryEvents", "query": None},#optional filter to subset events (e.g. only certain subjects or runs) — None means use everything
                "split": {"name": "SplitEvents", "val_ratio": 0.1},#splits events into train/val sets using val_ratio=0.1 (10% validation). This is a global split applied to all event types consistently — the split labels are stored in the events DataFrame and propagate through the whole pipeline
            },
        },
        "neuro": neuro_extractor,
        "features_to_use": ["text", "audio", "video"],# ist of modalities used during training
        "text_feature": text_feature,
        "video_feature": video_feature,
        "audio_feature": audio_feature,
        "image_feature": image_feature,# defined but never used for Algonaut study at least
        "batch_size": 8, # training batch size — number of fMRI segments per training step
    },
    "wandb_config": {
        "log_model": False,
        "entity": WANDB_ENTITY,
        "project": PROJECT_NAME,
        "group": "tribe_v2_test_run",
    },
    "brain_model_config": {
        "name": "FmriEncoder",
        "low_rank_head": 2048,# of the transformer encoder
        "hidden": 1152, # of the transformer encoder
        "extractor_aggregation": "cat",# how to combine text+audio+video features — concatenate
        "layer_aggregation": "cat",# how to combine multiple layers from each extractor — concatenate
        "combiner": None,
        "encoder": {
            "depth": 8,
        },
        "subject_layers": {"subject_dropout": 0.1},
        "subject_embedding": True,# default value was False
        "modality_dropout": 0.3,
    },
    "metrics": [
        {
            "log_name": "pearson",
            "name": "OnlinePearsonCorr",
            "dim": 0,
        },
        {
            "log_name": "subj_pearson",
            "name": "GroupedMetric",
            "metric_name": "OnlinePearsonCorr",
            "kwargs": {"dim": 0},
        },
        {
            "log_name": "retrieval_top1",
            "name": "TopkAcc",
            "topk": 1,
        },
    ],
    "loss": {"name": "MSELoss", "kwargs": {"reduction": "none"}},
    "optim": {
        "name": "LightningOptimizer",
        "optimizer": {
            "name": "Adam",
            "lr": 1e-4,
            "kwargs": {
                "weight_decay": 0.0,
            },
        },
        "scheduler": {
            "name": "OneCycleLR",
            "kwargs": {
                "max_lr": 1e-4,
                "pct_start": 0.1,
            },
        },
    },
    "n_epochs": 15,
    "limit_train_batches": None,# if set to e.g. 10, only runs 10 batches per epoch — useful for debugging. None = full dataset
    "patience": None,
    "enable_progress_bar": True,
    "log_every_n_steps": 5,
    "fast_dev_run": False,# PyTorch Lightning flag — if True, runs 1 batch of train+val then stops. For quick sanity checks
    "seed": 33,
}


if __name__ == "__main__":
    # The following can be used for local debugging/quick tests.

    from ..main import TribeExperiment

    exp = TribeExperiment(
        **default_config,
    )

    exp.infra.clear_job()
    out = exp.run()
    print(out)
