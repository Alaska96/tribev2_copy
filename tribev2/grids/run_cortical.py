# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from exca import ConfDict
from neuraltrain.utils import run_grid

from ..main import TribeExperiment  # type: ignore
from .configs import mini_config #  light configs for test only 
from .defaults import default_config 
from .configs import base_config # this is to select 20 compressed layers from each extractor
GRID_NAME = "tribe_v2_network_specific_multimodal"#"tribe_v2_basline_cortical"#"tribe_v2_network_specific_text_longcontext"#

update = {
    "wandb_config.group": GRID_NAME,
    "data.neuro.projection": None, # skip projection to use parcellated data instead
    #"infra.slurm_partition": "only-one-gpu", # fix null partition
    "infra.timeout_min": 60 * 24 * 2,          # 2 days ( QOS limit )
    #"brain_model_config.modality_dropout": 0.4,
    "networks_file": "/scratch_share/islab/Chaima/tribe_v2_work_space/schaefer1000_7net_networks.json",
    #"infra.workdir": None,# neuro data will not be cashed
    #"seed": None,  # random seed per model
    #"data.study.transforms.addcontext.max_context_len": None,***
    #"data.study.transforms.addepisodecontext": {"name": "AddEpisodeContextToWords"},#****
    #"data.text_feature.batch_size": 1,#***
}

#grid = {# contains grid of the tribev1
    #"data.study.names": ["Algonauts2025"], # Other studies can be added here to be considered # 1st point of coumunct with data
   # "data.features_to_use": [["text"]],
    #"target_network": ["Default", "Cont"],
    #"data.features_to_use": [["text", "audio", "video"]],
    #"target_network": ["Default", "Cont", "Vis", "DorsAttn"],
    #"data.features_to_use": [["audio", "video"]],
   # "target_network": ["Default", "Cont"],
   # "data.layers_to_use":[[0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1]],
    #"data.layers_to_use": [[0, 0.5, 1], [0.5, 0.75, 1.0], [0.5, 1], [0, 0.2, 0.4, 0.6, 0.8, 1]],
    #"loss.name": ["MSELoss", "SmoothL1Loss", "HuberLoss"],
    #"data.layer_aggregation": [None, "group_mean"],
    #"brain_model_config.subject_embedding": [True, False],
    #"brain_model_config.extractor_aggregation": ["cat", "sum", "stack"],
    #"brain_model_config.feature_aggregation": ["cat", "sum"],
    #"brain_model_config.modality_dropout": [0.0, 0.2, 0.4],
#}


grid = {
    "data.study.names": ["Algonauts2025"],
    "data.features_to_use": [["text"]],
    #"target_network": ["Default", "Cont"],
    "target_network": ["Default"],
    #"data.text_feature.model_name": ["meta-llama/Llama-3.2-1B"],
    #"data.text_feature.model_name": ["meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-1B"],
    #"seed": [33, 34, 35],
}
if __name__ == "__main__":
    #updated_config = ConfDict(default_config) # use default_config instead of mini_config
    updated_config = ConfDict(base_config) # use baseconfig
    updated_config.update(update)

    out = run_grid(
        TribeExperiment,
        GRID_NAME,
        updated_config,
        grid,
        job_name_keys=["wandb_config.name", "infra.job_name"],
        combinatorial=True, # can be set to False since for tribe v1 logic it  will use random sampling as set below
        #n_randomly_sampled=5,#### for tribe v1 logic "ensembling"
        overwrite=False,
        dry_run=False,#False
        infra_mode="retry", #resubmit crashed jobs considering cashes, while for #infra_mode="force" :jobs were cancelled before starting
    )
