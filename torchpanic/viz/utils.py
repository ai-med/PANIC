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
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..datamodule.modalities import ModalityType
from ..models.panic import PANIC

_NAME_MAPPER = None


@torch.no_grad()
def get_tabnet_predictions(model: PANIC, data_loader: DataLoader, device = torch.device("cuda")):
    model = model.eval().to(device)
    bias = model.nam.bias.detach().cpu().numpy()[np.newaxis]

    all_logits = []
    predictions = []
    for x, x_raw, aug, y in data_loader:
        logits, similarities, occurrences, nam_features_without_dropout = model(
            x[ModalityType.PET].to(device),
            x[ModalityType.TABULAR].to(device),
        )
        nam_features_without_dropout = nam_features_without_dropout.detach().cpu().numpy()
        outputs = np.concatenate((
            np.tile(bias, [logits.shape[0], 1, 1]),
            nam_features_without_dropout,
        ), axis=1)
        predictions.append(outputs)
        all_logits.append(logits.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits)
    y_pred = np.argmax(all_logits, axis=1)
    return np.concatenate(predictions), all_logits, y_pred


def _set_name_mapper(metadata: str):
    global _NAME_MAPPER

    name_mapper = {
        "real_age": "Age",
        "PTEDUCAT": "Education",
        "PTGENDER": "Male",
        "ABETA": "A$\\beta$",
        "TAU": "Tau",
        "PTAU": "p-Tau",
        "Left-Hippocampus": "Left Hippocampus",
        "Right-Hippocampus": "Right Hippocampus",
        "lh_entorhinal_thickness": "Left Entorhinal Cortex",
        "rh_entorhinal_thickness": "Right Entorhinal Cortex",
    }
    snp_metadata = pd.read_csv(
        metadata, index_col=0,
    )
    snp_key = snp_metadata.agg(
        lambda x: ":".join(map(str, x[
                ["Chromosome", "Position", "Allele2_reference", "Allele1_alternative"]
            ])) + "_" + x["Allele1_alternative"],
        axis=1
    )
    name_mapper.update((dict(zip(snp_key, snp_metadata.loc[:, "rsid"]))))
    _NAME_MAPPER = name_mapper


def _get_name_mapper(metadata: str):
    global _NAME_MAPPER

    if _NAME_MAPPER is None:
        _set_name_mapper(metadata)
    return _NAME_MAPPER


def map_tabular_names(names: Sequence[str], metadata_file: str) -> Sequence[str]:
    name_mapper = _get_name_mapper(metadata_file)
    return [name_mapper.get(name, name) for name in names]


def get_tabnet_output_names(
        tabular_names: Sequence[str], n_prototypes: int, config: Dict[str, Any], metadata_file: str
) -> Sequence[str]:
    name_mapper = _get_name_mapper(metadata_file)

    # reorder columns so they are in the same order as the output of the model
    nam_config = config["model"]["net"]["nam"]
    order = nam_config["idx_real_features"] + nam_config["idx_cat_features"]

    features_names = [name_mapper.get(tabular_names[i], tabular_names[i]) for i in order]

    prototype_names = [f"FDG-PET Proto {i}" for i in range(n_prototypes)]
    return ["bias"] + prototype_names + features_names
