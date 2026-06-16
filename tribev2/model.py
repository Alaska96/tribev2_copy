# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
* This script defines the architecture and forward pass for one single batch of data [ How is prediction for one batch of data performed?]
* Data preprocessing pipline afte the extractors phase
"""
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
    """ 
    * Config dataclass for 1D temporal smoohting filter  applied befor the transformer encoder
    * filter size =9
    * sigma=None --> use a uniform box ,esle use the gaussian_kernel_1d
    * This class is used optionally after features combination accross modalities, right befor transformer encoder
  """
    kernel_size: int = 9 # filter size = number of time points in the filter window
    sigma: float | None = None # if none--> use uniform box filter(learned) ; if set: fixed Guassian filter(frozen) "gaussian_kernel_1d"
  
    def build(self, dim: int) -> nn.Module:#builds a depthwise Conv1d smoothing filter for a feature dimension of size dim
    
        def gaussian_kernel_1d(kernel_size: int, sigma: float): # creates centered positions [-4,...,0,...,4] for a size-9 kernel
            x = torch.arange(kernel_size) - kernel_size // 2  # computes Gaussian bell curve at those positions
            kernel = torch.exp(-0.5 * (x / sigma) ** 2) # normalizes so weights sum to 1
            kernel = kernel / kernel.sum() # reshapes to [1, 1, kernel_size] as required by nn.Conv1d
            return kernel.view(1, 1, -1)
        # depthwise Conv1d: each channel filtered independently (groups=dim)
        conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2, #keeps time dimension length unchanged
            bias=False,# smoothing should not shift values
            groups=dim,
        )
        if self.sigma is not None: # build fixed Gaussian kernel and assign to conv weights
            kernel = gaussian_kernel_1d(kernel_size=self.kernel_size, sigma=self.sigma) # repeat kernel for each channel: [1,1,k] -> [dim,1,k]
            kernel = kernel.repeat(dim, 1, 1)
            conv.weight.data = kernel
            conv.requires_grad = False # freeze: Gaussian kernel is not learned
        return conv


class FmriEncoder(BaseModelConfig):
    """
       * Configuration class for FmriencoderModel
       * Defines the configuration for  the full trainable  data preprocessing & encoding pipeline: project each modality -> combine -> encode
    """

    # architecture
    projector: BaseModelConfig = Mlp(norm_layer="layer", activation_layer="gelu") # One MLP projector per modality ,to project modalities into a shared hidden-dimentional space
    combiner: Mlp | None = Mlp(norm_layer="layer", activation_layer="gelu") # single MLP applied after corss modal aggregation,learns cross-modal interaction per time step (at features dimension only) before feeding transformer   (I don't remember that this was in the desing they shared in the paper)??????????????????
    encoder: TransformerEncoder | None = TransformerEncoder() # transformer that learns temporal dependencies accross the contexualized multimodal embeddings 
    # other hyperparameters 
    time_pos_embedding: bool = True # if True--> add learnable positional embeddings along  time axis so that the transformer knows which timepoint is which 
    subject_embedding: bool = False # if True--> add per subject bias vector to embeddings before the transformer
    subject_layers: SubjectLayers | None = SubjectLayers() # subject-specific linear layers in the prediction head , allowing per subject outout adaptation
    hidden: int = 256 # feature vector size(embedding size)
    max_seq_len: int = 1024 # maximum number of time poitnts supported by the positional embedding table
    dropout: float = 0.0  # single dropout rate , propagated to projector MLPs and transfromer sub-components(attention, feedforward,layer dropout)
    extractor_aggregation: tp.Literal["stack", "sum", "cat"] = "cat" # different methods to apply cross-modal aggregration[cat=concatenate along feature axis, sum=element-wise addition, stack=concatenate along time axis] 
    layer_aggregation: tp.Literal["mean", "cat"] = "cat" # different methods to apply per modality  feature aggreagation (Extractor layer aggregation)[ mean=average across layers, cat=concatenate along feature axis]
    linear_baseline: bool = False # if True--> skip the transformer entirely, only subject-specific linear layers are used (for ablation)
    # regularization during training
    modality_dropout: float = 0.0 # per sample probability of not considering a modality(zeroing out an entire modality during training)
    temporal_dropout: float = 0.0 # per sample probability of not considering  time steps(zeroing out individual time steps during training)
    low_rank_head: int | None = None # if set --> bottleneck before the output head:compress hidden--> low_rank_head then predict [adds a bottleneck linear layer (hidden -> low_rank_head) before the prediction head to reduce parameter count]
    temporal_smoothing: TemporalSmoothing | None = None #if set: apply 1D smoothing filter to embeddings before the transformer
    def model_post_init(self, __context):
        """
        * Pydantic post-init hook called automatically after config creation
        * propagates the single dropout value to all sub-components so it only needs to be set once
        """
        if self.encoder is not None:
            for key in ["attn_dropout", "ff_dropout", "layer_dropout"]:
                setattr(self.encoder, key, self.dropout)
        if hasattr(self.projector, "dropout"):
            self.projector.dropout = self.dropout
        return super().model_post_init(__context)

    def build(# factory method: given data shape info, instantiates the actual nn.Module
    
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
    * The actual PyTorch model built from FmriEncoder config
    * Implements the full forward pass: feature aggregation -> smoothing -> combiner -> transformer -> prediction
   """
    def __init__( # construction phase no data flows yet 
        self,
        feature_dims: dict[str, tuple[int, int]], # # {modality_name: (num_layers, feature_dim)} — extractor output shape per modality
        n_outputs: int, # number of brain output units (vertices or voxels) to predict
        n_output_timesteps: int, # target number of output time steps after pooling the transformer output, to align with fMRI acquisition frequency
        config: FmriEncoder,
    ):
        super().__init__()
        self.config = config
        self.feature_dims = feature_dims # per modality or multimodel featues? # per-modality dictionary describing extractor output shapes
        self.n_outputs = n_outputs # for Fmri projector?--> yes for the predictor
        self.n_output_timesteps = n_output_timesteps # predictor's output number after pooling
        self.projectors = nn.ModuleDict() # holds one MLP projector per modality, registered as PyTorch submodules
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps) # applys resampling  on the predictor's output along time axis to match fMRI frequency
        hidden = config.hidden # the size of the embedding after going throught the preprocessing phase and is ready to be fed to the transformer 
        # the feature_dims below? is a dictionary passed when building the model . keys are modalities names("text", "audio", "video"). values are tuples (num_layers, feature_dim) describing the extractor output shape.
        for modality, tup in feature_dims.items():
            """
            *  Per-modality projector construction 
            *  Iterates over modalities, applies layer aggregation logic, and builds one MLP projector per modality
            """
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
        """
        * -------- Combiner construction ----------------
        * compute the exact input size to the combiner after cross-modal aggregation
        * uses integer division result (not just hidden) to avoid dimension mismatch from rounding
        """"
        input_dim = ( # define the input shape for the cross_modal combiner
            (hidden // len(feature_dims)) * len(feature_dims) # why not just hidden?--> may be slightly less than hidden due to integer division,because at the projector step we defined the output shape to be equal to hidden // len(feature_dims) where len(feature_dims) is the number of modalities --> since this is an integer devision it is not guaranteed that the sum of all modalities embeddings at same time point will be equal to exactly =hidden, thus the exact shape derivation is needed to avoid mismatch between the projector output shape and the combiner input shape since they are consecutive
            if config.extractor_aggregation == "cat"
            else hidden
        )
        if self.config.combiner is not None: #apply combiner
            self.combiner = self.config.combiner.build(input_dim, hidden) ## build MLP that learns cross-modality interactions in the feature dimension after aggregation
        else:# no combiner: assert hidden divides evenly so concatenated size already equals hidden exactly
            assert (
                hidden % len(feature_dims) == 0
            ), "hidden must be divisible by the number of modalities if there is no combiner"
            self.combiner = nn.Identity()## pass-through, no learned mixing
        # --- Optional low-rank bottleneck before prediction head ---
        if config.low_rank_head is not None:
            self.low_rank_head = nn.Linear(hidden, config.low_rank_head, bias=False)
            bottleneck = config.low_rank_head  # compressed size used as predictor input 
        else:
            bottleneck = hidden # no compression: use full hidden size
        """
        * ---------- Prediction head --------
        *subject-specific linear layers mapping from bottleneck to brain output units
        """ 
        self.predictor = config.subject_layers.build(
            in_channels=bottleneck, # pass the botteneck value to be the predictor input size
            out_channels=n_outputs,
        )
         # --- Optional temporal smoothing ---
        if config.temporal_smoothing is not None: # aapplied after cross-modal aggregation and before combinator and transformer
            self.temporal_smoothing = config.temporal_smoothing.build(dim=hidden)
        # --- Transformer and support embeddings (skipped in linear baseline mode) ---
        if not config.linear_baseline: # This is the case when transformer encoder is not skipped and thus deciding whether to add supprort embeddings based on the hyperparameters values
            if config.time_pos_embedding: # positional embeddings
                # learnable table of shape [1, max_seq_len, hidden]: one vector per time position
                self.time_pos_embed = nn.Parameter(
                    torch.randn(1, config.max_seq_len, hidden)
                
            if config.subject_embedding: # subject embeddings
                 # lookup table mapping subject id to a bias vector added to embeddings
                self.subject_embed = nn.Embedding(config.n_subjects, hidden)
            self.encoder = config.encoder.build(dim=hidden)
   # End of constraction phase 
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch: SegmentData, pool_outputs: bool = True) -> torch.Tensor: # # full forward pass: aggregates features, --> puts all pieces together
        x = self.aggregate_features(batch)  # B, T, H # this encapsulates all the embeddings prepocessing pipline and return embeddings of size = hidden ready to be fed to the transformer
        subject_id = batch.data.get("subject_id", None) # get traget subject id 
        if hasattr(self, "temporal_smoothing"): # temporal smoothing befor transformer and after cross-modal aggregation
            x = self.temporal_smoothing(x.transpose(1, 2)).transpose(1, 2)
        if not self.config.linear_baseline: # the transformer is not to be skipped
            x = self.transformer_forward(x, subject_id)
        x = x.transpose(1, 2)  # B, H, T
        if self.config.low_rank_head is not None:
            x = self.low_rank_head(x.transpose(1, 2)).transpose(1, 2)
        x = self.predictor(x, subject_id)  # B, O, T # subject prediction
        if pool_outputs: # temporal resampling(pooling)
            out = self.pooler(x)  # B, O, T' 
        else:
            out = x
        return out

    def aggregate_features(self, batch): # iterates over the batch data applying preprocessing to get embeddings at needed state to be passed to the transformer encoder [ note that the combiner step is included in the transformer forward function below
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
        # Apply cross_modal aggregation
        if self.config.extractor_aggregation == "stack":
            out = torch.cat(tensors, dim=1)
        elif self.config.extractor_aggregation == "cat":
            out = torch.cat(tensors, dim=-1)
        elif self.config.extractor_aggregation == "sum":
            out = sum(tensors)
        if self.config.temporal_dropout > 0 and self.training: # time points dropout
            for batch_idx in range(out.shape[0]):
                mask = torch.rand(out.shape[1]) < self.config.temporal_dropout
                out[batch_idx, mask, :] = torch.zeros_like(out[batch_idx, mask, :])
        return out

    def transformer_forward(self, x, subject_id=None): # forward for the transformer encoder only
        x = self.combiner(x)  # the cross_modal combination is performed here right before passing x to the the transformer
        if hasattr(self, "time_pos_embed"): # add positional embedding
            x = x + self.time_pos_embed[:, : x.size(1)]
        if hasattr(self, "subject_embed"): # add subject embedding
            x = x + self.subject_embed(subject_id)
        x = self.encoder(x) # transformer encoder 
        return x
