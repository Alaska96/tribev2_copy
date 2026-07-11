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
}

grid = {
    "data.study.names": ["Algonauts2025"], # Other studies can be added here to be considered 
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
        combinatorial=True,
        overwrite=False,
        dry_run=False,
        infra_mode="retry", #resubmit crashed jobs considering cashes, while for #infra_mode="force" :jobs were cancelled before starting
    )
