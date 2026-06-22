# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from exca import ConfDict
from neuraltrain.utils import run_grid

from ..main import TribeExperiment  # type: ignore
from .configs import mini_config

GRID_NAME = "tribe_v2_basline_parcelled"

update = {
    "wandb_config.group": GRID_NAME,
    #"data.study.names": "Algonauts2025",
    "data.neuro.projection": None, # skip projection step to use parcellated data instead
    #"data.neuro.infra.folder": None,  # fix fMRI cache warning
    #"data.neuro.infra.cluster": None, 
    #"infra.slurm_partition": "only-one-gpu", # fix null partition
    "infra.timeout_min": 60 * 24 * 2,          # 2 days
}

grid = {
    "data.study.names": ["Algonauts2025"],
}

if __name__ == "__main__":
    updated_config = ConfDict(mini_config)
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
        infra_mode="retry",#infra_mode="force"
    )
