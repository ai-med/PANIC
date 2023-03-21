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
from typing import Any, Dict
import torch

from .protowrapper import ProtoWrapper
from torchpanic.datamodule.adni import ModalityType


def flatten_module(module):
    children = list(module.children())
    flat_children = []
    if children == []:
        return module
    else:
        for child in children:
            try:
                flat_children.extend(flatten_module(child))
            except TypeError:
                flat_children.append(flatten_module(child))
    return flat_children


def deactivate_features(features):
    for param in features.parameters():
        param.requires_grad = False


class Encoder(torch.nn.Module):
    def __init__(
        self,
        protonet: Dict[Any, Any],
    ) -> None:
        super().__init__()
        wrapper = ProtoWrapper(
            **protonet
        )
        self.features = wrapper.features
        self.gap = torch.nn.AdaptiveAvgPool3d(1)
        out_chans_encoder = self.features[-1][-1].conv2.out_channels
        self.classification = torch.nn.Linear(out_chans_encoder, protonet['out_features'])
        self.nam_term_faker = None

    def forward(self, x):

        x = x[ModalityType.PET]
        out = self.features(x)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.classification(out)
        if self.nam_term_faker is None:
            self.nam_term_faker = torch.zeros_like(out).unsqueeze(-1)
        return out, self.nam_term_faker


