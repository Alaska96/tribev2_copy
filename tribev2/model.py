# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import torch
from einops import rearrange
from neuralset.dataloader import SegmentData
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.models.common import Mlp, SubjectLayers, SubjectLayersModel
from neuraltrain.models.transformer import TransformerEncoder
from torch import nn

logger = logging.getLogger(__name__)


class TemporalSmoothing(BaseModelConfig):
    kernel_size: int = 9
    sigma: float | None = None
  """ 
    * Config dataclass for 1D smoohting filter 
    * filter size =9
    * sigma=None --> use a uniform box ,esle use the gaussian_kernel_1d
    * This class is used optionally after features combination accross modalities, right befor transformer encoder
  """
    def build(self, dim: int) -> nn.Module:

        def gaussian_kernel_1d(kernel_size: int, sigma: float):
            x = torch.arange(kernel_size) - kernel_size // 2
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel = kernel / kernel.sum()
            return kernel.view(1, 1, -1)

        conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=False,
            groups=dim,
        )
        if self.sigma is not None:
            kernel = gaussian_kernel_1d(kernel_size=self.kernel_size, sigma=self.sigma)
            kernel = kernel.repeat(dim, 1, 1)
            conv.weight.data = kernel
            conv.requires_grad = False
        return conv


class FmriEncoder(BaseModelConfig):
    """
       * Configuration class for Fmriencoder
       * Porject each brain modality into a shared dimension
       * Combine them into one vector
       * Encode with the transformer
      
    """

    # architecture
    projector: BaseModelConfig = Mlp(norm_layer="layer", activation_layer="gelu") # One MLP per modality ,to project modalities into common linear space
    combiner: Mlp | None = Mlp(norm_layer="layer", activation_layer="gelu") # learns interaction cross modalities per time step level(on the features dimension only) befor feeding transformer   (I dont remember that this was in the desing they shared in the paper)??????????????????
    encoder: TransformerEncoder | None = TransformerEncoder() # learns temporal dependencies between the contexualized multimodal embeddings 
    # other hyperparameters 
    time_pos_embedding: bool = True # add learnable position embedding along with time axis so that the transformer knows which timepoint is which 
    subject_embedding: bool = False # add per subject bias vector 
    subject_layers: SubjectLayers | None = SubjectLayers() # subject_specific linear layers for the final prediction head , allowing per subject outout adaptation
    hidden: int = 256 #?????????????????????
    max_seq_len: int = 1024 # per modality embeddings length?????????
    dropout: float = 0.0  # ????????????????????
    extractor_aggregation: tp.Literal["stack", "sum", "cat"] = "cat" # different methods to merge features from different  modalities along time axis ???
    layer_aggregation: tp.Literal["mean", "cat"] = "cat" # how to merge features accross neural nework layers(extractor layers) of same modality ???
    linear_baseline: bool = False # if set to True--> the transformer will be entirely skipped , considering only liear model(subject linear layers)
    # Regularization during training to prevent overfitting
    modality_dropout: float = 0.0 # per sample probability of not considering a modality
    temporal_dropout: float = 0.0 # per sample probability of not considering a time step
    low_rank_head: int | None = None # optional bottleneck before the output head:compress hidden--> low_rank_head then predict
    temporal_smoothing: TemporalSmoothing | None = None # Optionally apply the smoothing filter defined above and before the transformer
    def model_post_init(self, __context):
        if self.encoder is not None:
            for key in ["attn_dropout", "ff_dropout", "layer_dropout"]:
                setattr(self.encoder, key, self.dropout)
        if hasattr(self.projector, "dropout"):
            self.projector.dropout = self.dropout
        return super().model_post_init(__context)

    def build(
        self, feature_dims: dict[int], n_outputs: int, n_output_timesteps: int
    ) -> nn.Module:
        return FmriEncoderModel(
            feature_dims,
            n_outputs,
            n_output_timesteps,
            config=self,
        )


class FmriEncoderModel(nn.Module):
   """
   * The main Encoder model
   * Uses configuration set in the configuration class FmriEncoder
   """
    def __init__( # construction phase no data flows yet 
        self,
        feature_dims: dict[str, tuple[int, int]],
        n_outputs: int,
        n_output_timesteps: int,
        config: FmriEncoder,
    ):
        super().__init__()
        self.config = config
        self.feature_dims = feature_dims # per modality or multimodel featues?
        self.n_outputs = n_outputs # for Fmri projector?--> yes for the predictor
        self.n_output_timesteps = n_output_timesteps # transformer's output frequency ?
        self.projectors = nn.ModuleDict() # dictionary to hold results of modalities projection?
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps) # applied on the transformers output to resample stimuli to match fmri frequency
        hidden = config.hidden # the size of the embedding after going throught the preprocessing phase and is ready to be fed to the transformer 
        # the feature_dims below? is a dictionary passed when building the model . keys are modalities names("text", "audio", "video"). values are tuples (num_layers, feature_dim) describing the extractor output shape.
        for modality, tup in feature_dims.items(): #this loop, iterates along modalities time series [Lm,Dm] and passes them through: per modality layer agregation--> per modality linear projection and dim reduction 
            #modality — the current modality name, e.g. "text".
            #tup — the value for that modality, e.g. (32, 2048) for Llama with 32 layers and dim 2048. Can be None if that modality has no data.
            #num_layers — L, number of extractor layers for this modality.
            #feature_dim — D, dimensionality of each layer's embedding.
            if tup is None:
                logger.warning(
                    "%s has no feature dimensions. Skipping projector.", modality
                )
                continue
            else:
                num_layers, feature_dim = tup # from tup extract num_layers=Lm and feature_dim=Dm ,[Lm,Dm]--> this represents the shape of one embedding of each feature extracctor where Lm: is the number of layers of the extracctor and Dm is its dimension(embedding length)
            input_dim = ( #This is the size of the vector that enters the projector for this modality
                feature_dim * num_layers # if cat is the selected layer agregation method , the embeddings will be of shape [1,Lm*Dm]
                if config.layer_aggregation == "cat"# note that input dim is controled by layer_aggregation choice
                else feature_dim # if instead mean is selected as aggregation methode the embedding will have size Dm=feature_dim
            )
            output_dim = ( # defines the input shape of the modality projector
                hidden // len(feature_dims)  # this allows fair contribution of all modalities and ensures that the final embedding shape after cross modal aggregation is exactly =hidden
                if config.extractor_aggregation == "cat" # note that different from input_dim, output_dim is controlled by extractor_aggregation method
                else hidden
            )
            self.projectors[modality] = self.config.projector.build( # apply projection per modality using the derived input_dim and output_dim
                input_dim, output_dim
            )
        # End of For the FOR loop above, at this point we have all modalities projected into same linear space and have same dimension= hidden//number of modalities if cat is chosen as cross_modal aggregation method,else dinm=hidden , next step is --> cross modal agregation --> combination
        input_dim = ( # define the input shape for the cross_modal combiner
            (hidden // len(feature_dims)) * len(feature_dims) # why not just hidden?--> because at the projector step we defined the output shape to be equal to hidden // len(feature_dims) where len(feature_dims) is the number of modalities --> since this is an integer devision it is not guaranteed that the sum of all modalities embeddings at same time point will be equal to exactly =hidden, thus the exact shape derivation is needed to avoid mismatch between the projector output shape and the combiner input shape since they are consecutive
            if config.extractor_aggregation == "cat"
            else hidden
        )
        if self.config.combiner is not None: # apply combiner 
            self.combiner = self.config.combiner.build(input_dim, hidden)
        else:
            assert (
                hidden % len(feature_dims) == 0
            ), "hidden must be divisible by the number of modalities if there is no combiner"
            self.combiner = nn.Identity()
        # This part is responsible of deciding which value to select for bottelneck at predictor input level based on the value of confi.low_rank_head hyperparameter set at configuration level
        if config.low_rank_head is not None:
            self.low_rank_head = nn.Linear(hidden, config.low_rank_head, bias=False)
            bottleneck = config.low_rank_head
        else:
            bottleneck = hidden # else use the default value?
        self.predictor = config.subject_layers.build(
            in_channels=bottleneck, # pass the botteneck value to be the predictor input size
            out_channels=n_outputs,
        )
        if config.temporal_smoothing is not None: # temporal pooling is applied on transformers outputs to resample stimuli to be aligned with fmri frequency 
            self.temporal_smoothing = config.temporal_smoothing.build(dim=hidden)
        if not config.linear_baseline: # This is the case when transformer encoder is not skipped and thus deciding whether to add supprort embeddings based on the hyperparameters values
            if config.time_pos_embedding: # positional embeddings
                self.time_pos_embed = nn.Parameter(
                    torch.randn(1, config.max_seq_len, hidden)
                )
            if config.subject_embedding: # subject embeddings
                self.subject_embed = nn.Embedding(config.n_subjects, hidden)
            self.encoder = config.encoder.build(dim=hidden)
   # End of constraction phase 
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch: SegmentData, pool_outputs: bool = True) -> torch.Tensor: # forward pass --> puts all pieces together
        x = self.aggregate_features(batch)  # B, T, H # this encapsulates all the embeddings prepocessing pipline and return embeddings of size = hidden ready to be fed to the transformer
        subject_id = batch.data.get("subject_id", None) #???
        if hasattr(self, "temporal_smoothing"): # temporal smoothing befor transformer, why?
            x = self.temporal_smoothing(x.transpose(1, 2)).transpose(1, 2)
        if not self.config.linear_baseline: # When it is chosen that the transformer is not to be skipped
            x = self.transformer_forward(x, subject_id)
        x = x.transpose(1, 2)  # B, H, T
        if self.config.low_rank_head is not None:
            x = self.low_rank_head(x.transpose(1, 2)).transpose(1, 2)
        x = self.predictor(x, subject_id)  # B, O, T
        if pool_outputs:
            out = self.pooler(x)  # B, O, T'
        else:
            out = x
        return out

    def aggregate_features(self, batch):
        tensors = []
        # get B, T
        for modality in batch.data.keys():
            if modality in self.feature_dims:
                break
        x = batch.data[modality]
        B, T = x.shape[0], x.shape[-1]
        for modality in self.feature_dims.keys():
            if modality not in self.projectors or modality not in batch.data:
                data = torch.zeros(
                    B, T, self.config.hidden // len(self.feature_dims)
                ).to(x.device)
            else:
                data = batch.data[modality]  # B, L, H, T
                data = data.to(torch.float32)
                if data.ndim == 3:
                    data = data.unsqueeze(1)
                # mean over layers
                if self.config.layer_aggregation == "mean":
                    data = data.mean(dim=1)
                elif self.config.layer_aggregation == "cat":
                    data = rearrange(data, "b l d t -> b (l d) t")
                data = data.transpose(1, 2)
                assert data.ndim == 3  # B, T, D
                if isinstance(self.projectors[modality], SubjectLayersModel):
                    data = self.projectors[modality](
                        data.transpose(1, 2), batch.data["subject_id"]
                    ).transpose(1, 2)
                else:
                    data = self.projectors[modality](data)  # B, T, H
                if self.config.modality_dropout > 0 and self.training:
                    mask = torch.rand(data.shape[0]) < self.config.modality_dropout
                    data[mask, :] = torch.zeros_like(data[mask, :])
            tensors.append(data)
        if self.config.extractor_aggregation == "stack":
            out = torch.cat(tensors, dim=1)
        elif self.config.extractor_aggregation == "cat":
            out = torch.cat(tensors, dim=-1)
        elif self.config.extractor_aggregation == "sum":
            out = sum(tensors)
        if self.config.temporal_dropout > 0 and self.training:
            for batch_idx in range(out.shape[0]):
                mask = torch.rand(out.shape[1]) < self.config.temporal_dropout
                out[batch_idx, mask, :] = torch.zeros_like(out[batch_idx, mask, :])
        return out

    def transformer_forward(self, x, subject_id=None):
        x = self.combiner(x)
        if hasattr(self, "time_pos_embed"):
            x = x + self.time_pos_embed[:, : x.size(1)]
        if hasattr(self, "subject_embed"):
            x = x + self.subject_embed(subject_id)
        x = self.encoder(x)
        return x
