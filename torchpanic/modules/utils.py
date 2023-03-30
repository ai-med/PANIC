# This file is part of Prototypical Additive Neural Network for Interpretable Classification (PANIC).
#
# PANIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PANIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PANIC. If not, see <https://www.gnu.org/licenses/>.
import math
from pathlib import Path
from typing import Union

from hydra.utils import instantiate as hydra_init
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from yaml import safe_load as yaml_load

import subprocess


def format_float_to_str(x: float) -> str:
    return "{:2.1f}".format(x * 100)


def init_vector_normal(vector: torch.Tensor):
    stdv = 1. / math.sqrt(vector.size(1))
    vector.data.uniform_(-stdv, stdv)


def get_current_stage(epoch, epochs_all, epochs_nam, warmup=10):
    total = epochs_all + epochs_nam
    stage = "nam_only"
    if epoch < 0.5 * warmup:
        stage = "warmup"
    elif epoch < warmup:
        stage = "warmup_protonet"
    elif (epoch - warmup) % total < epochs_all:
        stage = "all"
    return stage


def get_last_valid_checkpoint(path: Path):
    epoch = int(path.stem.split("=")[1].split("-")[0])
    epoch_old = int(path.stem.split("=")[1].split("-")[0])
    config = load_config(path)
    e_all = config.model.epochs_all
    e_nam = config.model.epochs_nam
    warmup = config.model.epochs_warmup

    stage = get_current_stage(epoch, e_all, e_nam, warmup)
    while stage == "nam_only":
        epoch -= 1
        stage = get_current_stage(epoch, e_all, e_nam, warmup)
    if epoch != epoch_old:
        epoch += 1
    ckpt_path = str(path)
    print(f"Previous epoch {epoch_old} was invalid. Valid checkpoint is of epoch {epoch}")
    return ckpt_path.replace(f"epoch={epoch_old}", f"epoch={epoch}")


def init_vectors_orthogonally(vector: torch.Tensor, n_protos_per_class: int):
    # vector has shape (n_protos, n_chans)
    assert vector.size(0) % n_protos_per_class == 0
    torch.nn.init.xavier_uniform_(vector)

    for j in range(vector.size(0)):
        vector.data[j, j // n_protos_per_class] += 1.


def load_config(ckpt_path: Union[str, Path]) -> DictConfig:
    config_path = str(Path(ckpt_path).parent.parent / '.hydra' / 'config.yaml')
    with open(config_path) as f:
        y = yaml_load(f)
    workdir = Path().absolute()
    idx = workdir.parts.index('PANIC')
    workdir = Path(*workdir.parts[:idx+1])
    if 'protonet' in y['model']['net']:
        y['model']['net']['protonet']['pretrained_model'] = \
            y['model']['net']['protonet']['pretrained_model'].replace('${hydra:runtime.cwd}', str(workdir))
    config = OmegaConf.create(y)
    return config


def load_model_and_data(ckpt_path: str, device=torch.device("cuda")):
    ''' loads model and data with respect to a checkpoint path
        must call data.setup(stage) to setup data
        pytorch model can be retrieved with model.net '''
    config = load_config(Path(ckpt_path))
    data = hydra_init(config.datamodule)
    model = hydra_init(config.model)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    return model, data, config


def get_git_hash():
    return subprocess.check_output([
        "git", "rev-parse", "HEAD"
    ], encoding="utf8").strip()
