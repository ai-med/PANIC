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
import torch

from .ppnet import PPNet

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


class ProtoWrapper(PPNet):
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
        super().__init__(
                backbone,
                in_channels,
                out_features,
                n_prototypes_per_class,
                n_chans_protos,
                optim_features,
                normed_prototypes,
                **kwargs)

    def base_forward(self, x):

        x = self.features(x)
        occurrences = self.occurrence_module(x)  # (bs, n_prototypes)
        feature_map = self.add_on_layers(x)      # (bs, n_chans_proto, h, w[, d])

        # the following should work for 3D and 2D data!
        broadcasting_shape = (occurrences.size(0), occurrences.size(1), feature_map.size(1), *feature_map.size()[2:])
        # of shape (bs, n_protos, n_chans_per_prot, h, w[, d])

        # expand the two such that broadcasting is possible, i.e. vectorization of prototype feature calculation
        occurrences_reshaped = occurrences.unsqueeze(2).expand(broadcasting_shape)
        feature_map_reshaped = feature_map.unsqueeze(1).expand(broadcasting_shape)

        # element-wise multiplication of each occurence map with the featuremap
        feature_vectors = occurrences_reshaped * feature_map_reshaped
        feature_vectors = feature_vectors.mean(dim=self.gap)  # essentially GAP over the spatial resolutions
        # feature_vectors size is now (bs, n_protos, n_chans_per_prot)
        # prototype_vectors size is (n_protos, n_chans_per_prot)
        if self.normed_prototypes:
            feature_vectors = (feature_vectors / torch.linalg.vector_norm(
                feature_vectors, ord=2, dim=2, keepdim=True, dtype=torch.double)).to(torch.float32)

        # make prototypes broadcastable to featuer vectors
        similarities = self.cosine_similarity(feature_vectors, self.prototype_vectors.unsqueeze(0))
        return feature_vectors, similarities, occurrences

    def forward(self, x):

        _, similarities, occurrences = self.base_forward(x)

        logits = self.classification(similarities)

        return logits, similarities, occurrences

    @torch.no_grad()
    def push_forward(self, x):

        return self.base_forward(x)
