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
from torchvision import models

from .protowrapper import ProtoWrapper
from .nam import BaseNAM
from ..models.backbones import ThreeDResNet

BACKBONES = {
    'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
    'resnet152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
    '3dresnet': (ThreeDResNet, None),
}


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


class PANIC(ProtoWrapper):
    def __init__(
        self,
        protonet: Dict[Any, Any],
        nam: Dict[Any, Any],
    ) -> None:
        super().__init__(
            **protonet
        )
        # must tidy up prototype vector dimensions!
        self.classification = None
        self.nam = BaseNAM(
            **nam
        )

    def forward_image(self, image):

        return self.base_forward(image)

    def forward(self, image, tabular):

        feature_vectors, similarities, occurrences = self.forward_image(image)
        # feature_vectors shape is (bs, n_protos, n_chans_per_prot)
        # similarities is of shape (bs, n_protos)
        similarities_reshaped = similarities.view(
            similarities.size(0), self.num_classes, self.n_prototypes_per_class)
        similarities_reshaped = similarities_reshaped.permute(0, 2, 1)
        # this maps similarities such that we have the similarities w.r.t. each class:
        # new shape is (bs, n_protos_per_class, n_classes)

        nam_features = self.nam.base_forward(tabular)
        # nam_features shape is (bs, n_features, n_classes)

        features = torch.cat((similarities_reshaped, nam_features), dim=1)

        logits = self.nam.feature_dropout(features)
        logits = torch.sum(logits, dim=1) + self.nam.bias
        return logits, similarities, occurrences, features

    @torch.no_grad()
    def push_forward(self, image, tabular):

        return self.forward_image(image)
