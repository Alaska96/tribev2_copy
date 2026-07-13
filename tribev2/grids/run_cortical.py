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
GRID_NAME = "tribe_v2_basline_cortical"

update = {
    "wandb_config.group": GRID_NAME,
    "data.neuro.projection": None, # skip projection to use parcellated data instead
    #"infra.slurm_partition": "only-one-gpu", # fix null partition
    "infra.timeout_min": 60 * 24 * 2,          # 2 days ( QOS limit )
    "infra.workdir": None,
    "seed": None,  # random seed per model
}

grid = {# contains grid of the tribev1
    "data.study.names": ["Algonauts2025"], # Other studies can be added here to be considered 
    "data.layers_to_use": [[0, 0.5, 1], [0.5, 0.75, 1.0], [0.5, 1.], [0, 0.2, 0.4, 0.6, 0.8, 1.]],
    "loss.name": ["MSELoss", "SmoothL1Loss", "HuberLoss"],
    "data.layer_aggregation": [None, "group_mean"],
    "brain_model_config.subject_embedding": [True, False],
    "brain_model_config.extractor_aggregation": ["cat", "sum", "stack"],
    #"brain_model_config.feature_aggregation": ["cat", "sum"],
    "brain_model_config.modality_dropout": [0.0, 0.2, 0.4],
}

if __name__ == "__main__":
    updated_config = ConfDict(default_config) # use default_config instead of mini_config
    updated_config.update(update)

    out = run_grid(
        TribeExperiment,
        GRID_NAME,
        updated_config,
        grid,
        job_name_keys=["wandb_config.name", "infra.job_name"],
        combinatorial=True, # can be set to False since for tribe v1 logic it  will use random sampling as set below
        n_randomly_sampled=1000 #### for tribe v1 logic "ensembling"
        overwrite=False,
        dry_run=False,
        infra_mode="retry", #resubmit crashed jobs considering cashes, while for #infra_mode="force" :jobs were cancelled before starting
    )
