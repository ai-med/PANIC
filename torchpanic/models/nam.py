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
from typing import Dict, Sequence, Tuple
import torch
from torch import nn
from torch.nn import init

from ..datamodule.adni import ModalityType
from torchpanic.modules.utils import init_vector_normal


class ExU(nn.Module):
    """exp-centered unit"""
    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((1, out_features)))
        self.bias = nn.Parameter(torch.empty((1, out_features)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weight, mean=4.0, std=0.5)
        init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[1] == 1
        out = torch.exp(self.weight) * (x - self.bias)
        return out


class ReLUN(nn.Module):
    def __init__(self, n: float = 1.0) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0, max=self.n)


class FeatureNet(nn.Module):
    """A neural network for a single feature"""
    def __init__(
        self,
        out_features: int,
        hidden_units: Sequence[int],
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        in_features = hidden_units[0]
        layers = {
            "in": nn.Sequential(
                nn.utils.weight_norm(nn.Linear(1, in_features)),
                nn.ReLU()),
        }
        for i, units in enumerate(hidden_units[1:]):
            layers[f"dense_{i}"] = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_features, units)),
                nn.Dropout(p=dropout_rate),
                nn.ReLU(),
            )
            in_features = units
        layers["dense_out"] = nn.utils.weight_norm(nn.Linear(in_features, out_features, bias=False))
        self.hidden_layers = nn.ModuleDict(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.hidden_layers.values():
            out = layer(out)
        return out


class NAM(nn.Module):
    """Neural Additive Model

    .. [1] Neural Additive Models: Interpretable Machine Learning with Neural Nets. NeurIPS 2021
           https://proceedings.neurips.cc/paper/2021/hash/251bd0442dfcc53b5a761e050f8022b8-Abstract.html
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_units: Sequence[int],
        dropout_rate: float = 0.5,
        feature_dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        layers = {}
        for i in range(in_features):
            layers[f"fnet_{i}"] = nn.Sequential(
                FeatureNet(
                    out_features=out_features,
                    hidden_units=hidden_units,
                    dropout_rate=dropout_rate,
                ),
            )
        self.feature_nns = nn.ModuleDict(layers)
        self.feature_dropout = nn.Dropout1d(p=feature_dropout_rate)

        self.bias = nn.Parameter(torch.empty((1, out_features)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.bias)

    def base_forward(self, tabular: torch.Tensor) -> torch.Tensor:
        values, is_missing = torch.unbind(tabular, axis=1)

        # FIXME: Treat missing value in a better way
        x = values * (1.0 - is_missing)

        features = torch.split(x, 1, dim=-1)
        outputs = []
        for x_i, layer in zip(features, self.feature_nns.values()):
            outputs.append(layer(x_i))
        outputs = torch.stack(outputs, dim=1)
        logits = self.feature_dropout(outputs)
        return logits, outputs

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        tabular = batch[ModalityType.TABULAR]
        logits, outputs = self.base_forward(tabular)
        logits = torch.sum(logits, dim=1) + self.bias
        return logits, outputs


def is_unique(x):
    return len(x) == len(set(x))


class SemiParametricNAM(NAM):
    def __init__(
        self,
        idx_real_features: Sequence[int],
        idx_cat_features: Sequence[int],
        idx_real_has_missing: Sequence[int],
        idx_cat_has_missing: Sequence[int],
        out_features: int,
        hidden_units: Sequence[int],
        dropout_rate: float = 0.5,
        feature_dropout_rate: float = 0.5,
    ) -> None:
        super().__init__(
            in_features=len(idx_real_features),
            out_features=out_features,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            feature_dropout_rate=feature_dropout_rate,
        )

        assert is_unique(idx_real_features)
        assert is_unique(idx_cat_features)
        assert is_unique(idx_real_has_missing)
        assert is_unique(idx_cat_has_missing)

        self.cat_linear = nn.Linear(len(idx_cat_features), out_features, bias=False)
        n_missing = len(idx_real_has_missing) + len(idx_cat_has_missing)
        self.miss_linear = nn.Linear(n_missing, out_features, bias=False)

        self._idx_real_features = idx_real_features
        self._idx_cat_features = idx_cat_features
        self._idx_real_has_missing = idx_real_has_missing
        self._idx_cat_has_missing = idx_cat_has_missing

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        tabular = batch[ModalityType.TABULAR]
        values, is_missing = torch.unbind(tabular, axis=1)

        val_categ = values[:, self._idx_cat_features]
        miss_categ = is_missing[:, self._idx_cat_features]
        has_miss_categ = miss_categ[:, self._idx_cat_has_missing]

        val_real = values[:, self._idx_real_features]
        miss_real = is_missing[:, self._idx_real_features]
        has_miss_real = miss_real[:, self._idx_real_has_missing]

        out_real_full = self.forward_real(val_real) * torch.unsqueeze(1.0 - miss_real, dim=-1)
        out_real = torch.sum(out_real_full, dim=1)

        out_categ = self.cat_linear(val_categ * (1.0 - miss_categ))

        out_miss = self.miss_linear(torch.cat((has_miss_categ, has_miss_real), dim=1))

        return sum((out_real, out_categ, out_miss, self.bias,)), out_real_full

    def forward_real(self, x):
        features = torch.split(x, 1, dim=-1)
        outputs = []
        for x_i, layer in zip(features, self.feature_nns.values()):
            outputs.append(layer(x_i))

        outputs = torch.stack(outputs, dim=1)
        outputs = self.feature_dropout(outputs)
        return outputs


class BaseNAM(NAM):
    def __init__(
        self,
        idx_real_features: Sequence[int],
        idx_cat_features: Sequence[int],
        idx_real_has_missing: Sequence[int],
        idx_cat_has_missing: Sequence[int],
        out_features: int,
        hidden_units: Sequence[int],
        dropout_rate: float = 0.5,
        feature_dropout_rate: float = 0.5,
        **kwargs
    ) -> None:
        super().__init__(
            in_features=len(idx_real_features),
            out_features=out_features,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            feature_dropout_rate=feature_dropout_rate,
        )

        assert is_unique(idx_real_features)
        assert is_unique(idx_cat_features)
        assert is_unique(idx_real_has_missing)
        assert is_unique(idx_cat_has_missing)

        n_missing = len(idx_real_has_missing) + len(idx_cat_has_missing)
        self.tab_missing_embeddings = nn.Parameter(
            torch.empty((n_missing, out_features), dtype=torch.float32), requires_grad=True)
        nn.init.xavier_uniform_(self.tab_missing_embeddings)

        self.cat_linear = nn.Parameter(
            torch.empty((len(idx_cat_features), out_features), dtype=torch.float32, requires_grad=True))
        nn.init.xavier_uniform_(self.cat_linear)

        self._idx_real_features = idx_real_features
        self._idx_cat_features = idx_cat_features
        self._idx_real_has_missing = idx_real_has_missing
        self._idx_cat_has_missing = idx_cat_has_missing

    def forward_real(self, x):
        features = torch.split(x, 1, dim=-1)
        outputs = []
        for x_i, layer in zip(features, self.feature_nns.values()):
            outputs.append(layer(x_i))

        outputs = torch.stack(outputs, dim=1)
        return outputs

    def base_forward(self, tabular: torch.Tensor) -> torch.Tensor:
        values, is_missing = torch.unbind(tabular, dim=1)

        val_real = values[:, self._idx_real_features]
        miss_real = is_missing[:, self._idx_real_features]
        has_miss_real = miss_real[:, self._idx_real_has_missing]
        # TODO for all miss_real==1, check that its index is in self._idx_real_has_missing

        val_categ = values[:, self._idx_cat_features]
        miss_categ = is_missing[:, self._idx_cat_features]
        has_miss_categ = miss_categ[:, self._idx_cat_has_missing]
        # TODO for all miss_categ==1, check that its index is in self._idx_categ_has_missing

        features_real = self.forward_real(val_real)
        features_categ = self.cat_linear.unsqueeze(0) * val_categ.unsqueeze(-1)

        features_real = features_real * (1.0 - miss_real.unsqueeze(-1))  # set features to zero where they are mising
        features_categ = features_categ * (1.0 - miss_categ.unsqueeze(-1))

        filler_real = torch.zeros_like(features_real)
        filler_categ = torch.zeros_like(features_categ)

        filler_real[:, self._idx_real_has_missing, :] = \
            self.tab_missing_embeddings[len(self._idx_cat_has_missing):].unsqueeze(0) * has_miss_real.unsqueeze(-1)
        filler_categ[:, self._idx_cat_has_missing, :] = \
            self.tab_missing_embeddings[:len(self._idx_cat_has_missing)].unsqueeze(0) * has_miss_categ.unsqueeze(-1)

        features_real = features_real + filler_real  # filler only has values where real is 0
        features_categ = features_categ + filler_categ

        return torch.cat((features_real, features_categ), dim=1)

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        if isinstance(tabular, dict):
            tabular = tabular[ModalityType.TABULAR]
        features = self.base_forward(tabular)
        return torch.sum(self.feature_dropout(features), dim=1) + self.bias, features
