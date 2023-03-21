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
#
#
#
#
# This file is part of Dynamic Affine Feature Map Transform (DAFT).
#
# DAFT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DAFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DAFT. If not, see <https://www.gnu.org/licenses/>.
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from ...datamodule.modalities import ModalityType
from .vol_blocks import ConvBnReLU, DAFTBlock, ResBlock


class HeterogeneousResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        n_outputs: int = 3,
        bn_momentum: int = 0.1,
        n_basefilters: int = 4,
    ) -> None:
        super().__init__()

        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

    def forward(self, batch):
        image = batch[self.image_modality]

        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        # cannot return None, because lightning complains
        terms = torch.zeros((out.shape[0], 1, out.shape[1],), device=out.device)
        return out, terms


class DAFT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_tabular: int,
        n_outputs: int,
        idx_tabular_has_missing: Sequence[int],
        bn_momentum: float = 0.05,
        n_basefilters: int = 4,
        filmblock_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if filmblock_args is None:
            filmblock_args = {}

        self.image_modality = ModalityType.PET

        if min(idx_tabular_has_missing) < 0:
            raise ValueError("idx_tabular_has_missing contains negative values")
        if max(idx_tabular_has_missing) >= in_tabular:
            raise ValueError("index in idx_tabular_has_missing is out of range")
        idx_missing = frozenset(idx_tabular_has_missing)
        if len(idx_tabular_has_missing) != len(idx_missing):
            raise ValueError("idx_tabular_has_missing contains duplicates")

        self.register_buffer(
            'idx_tabular_has_missing',
            torch.tensor(idx_tabular_has_missing, dtype=torch.long),
        )
        self.register_buffer(
            'idx_tabular_without_missing',
            torch.tensor(list(set(range(in_tabular)).difference(idx_missing)), dtype=torch.long),
        )

        n_missing = len(idx_tabular_has_missing)
        self.tab_missing_embeddings = nn.Parameter(
            torch.empty((1, n_missing,), dtype=torch.float32), requires_grad=True)
        nn.init.xavier_uniform_(self.tab_missing_embeddings)

        self.split_size = 4 * n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(3, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = DAFTBlock(
            4 * n_basefilters,
            8 * n_basefilters,
            bn_momentum=bn_momentum,
            ndim_non_img=in_tabular,
            **filmblock_args,
        )  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

    def forward(self, batch):
        image = batch[self.image_modality]

        tabular_data = batch[ModalityType.TABULAR]
        values, is_missing = torch.unbind(tabular_data, axis=1)

        values_wo_missing = values[:, self.idx_tabular_without_missing]
        values_w_missing = values[:, self.idx_tabular_has_missing]
        missing_in_batch = is_missing[:, self.idx_tabular_has_missing]
        tabular_masked = torch.where(
            missing_in_batch == 1.0, self.tab_missing_embeddings, values_w_missing
        )

        features = torch.cat(
            (values_wo_missing, tabular_masked), dim=1,
        )

        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, features)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        # cannot return None, because lightning complains
        terms = torch.zeros((out.shape[0], 1, out.shape[1],), device=out.device)
        return out, terms
