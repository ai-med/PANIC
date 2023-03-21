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
# Code has been adapted from https://github.com/cfchen-duke/ProtoPNet, which
# has been released under the following license:
# MIT License
# 
# Copyright (c) 2019 Chaofan Chen (cfchen-duke), Oscar Li (OscarcarLi),
# Chaofan Tao, Alina Jade Barnett, Cynthia Rudin
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
from torchvision import models

from .backbones import ThreeDResNet

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


class PPNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        in_channels: int,
        out_features: int,
        n_prototypes_per_class: int,
        n_chans_protos: int,
        optim_features: bool,
        normed_prototypes: bool,
        **kwargs,
    ) -> None:
        super().__init__()
        assert backbone in BACKBONES.keys(), f"cannot find backbone {backbone} in valid BACKBONES {BACKBONES.keys()}!"

        self.normed_prototypes = normed_prototypes
        self.n_prototypes_per_class = n_prototypes_per_class

        features, weights = BACKBONES[backbone]
        if backbone.startswith("resnet"):
            weights = weights.DEFAULT
        if weights is None:
            if 'pretrained_model' in kwargs:
                pretrained_model = kwargs.pop('pretrained_model')
            else:
                pretrained_model = None
            weights = kwargs
            weights = kwargs
            weights["in_channels"] = in_channels
            weights["n_outputs"] = out_features
        else:
            weights = {"weights": weights}
        features = features(**weights)
        features = list(features.children())
        fc_layer = features[-1]
        features = features[:-2]  # remove GAP and last fc layer!
        in_layer = features[0]
        assert isinstance(in_layer, nn.Conv2d) or isinstance(in_layer, nn.Conv3d)
        self.two_d = isinstance(in_layer, nn.Conv2d)
        self.conv_func = nn.Conv2d if self.two_d else nn.Conv3d
        if in_layer.in_channels != in_channels:
            assert optim_features, "Different num input channels -> must optim!"
            in_layer = self.conv_func(
                in_channels=in_channels,
                out_channels=in_layer.out_channels,
                kernel_size=in_layer.kernel_size,
                stride=in_layer.stride,
                padding=in_layer.padding,
            )
            in_layer.requires_grad = False
            features[0] = in_layer
        self.num_classes = out_features
        self.n_features_encoder = fc_layer.in_features
        self.prototype_shape = (n_prototypes_per_class * self.num_classes, n_chans_protos)
        self.num_prototypes = self.prototype_shape[0]

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert (self.num_prototypes % self.num_classes == 0), \
            f"{self.num_prototypes} vs {self.num_classes}"  # not needed as we initialize differently
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = nn.Parameter(
            torch.zeros(self.num_prototypes, self.num_classes),
            requires_grad=False)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.features = torch.nn.Sequential(*features)

        self.add_on_layers = nn.Sequential(
            self.conv_func(in_channels=self.n_features_encoder, out_channels=self.prototype_shape[1], kernel_size=1),
            nn.ReLU(),
            self.conv_func(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
            nn.Softplus()
            )
        self.occurrence_module = nn.Sequential(
            self.conv_func(in_channels=self.n_features_encoder, out_channels=self.n_features_encoder // 8, kernel_size=1),
            nn.ReLU(),
            self.conv_func(in_channels=self.n_features_encoder // 8, out_channels=self.prototype_shape[0], kernel_size=1),
            nn.Sigmoid()
        )
        self.gap = (-2, -1) if self.two_d else (-3, -2, -1)  # if 3D network, pool h,w and d
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        if self.normed_prototypes:
            self.prototype_vectors.data = (self.prototype_vectors.data / torch.linalg.vector_norm(
                self.prototype_vectors.data, ord=2, dim=1, keepdim=True, dtype=torch.double)).to(torch.float32)
        # nn.init.xavier_uniform_(self.prototype_vectors)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        if pretrained_model is not None:
            state_dict = torch.load(pretrained_model, map_location=self.prototype_vectors.device)['state_dict']
            state_dict = {x.replace('net.features.', ''): y for x, y in state_dict.items()}
            state_dict = {x: y for x, y in state_dict.items() if x in self.features.state_dict().keys()}
            self.features.load_state_dict(state_dict)
        if not optim_features:
            for x in self.features.parameters():
                x.requires_grad = False

        self.classification = nn.Linear(self.num_prototypes, self.num_classes,
                                        bias=False)  # do not use bias

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.classification.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)
