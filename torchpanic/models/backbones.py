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
from collections import OrderedDict
from torch import nn


def conv3d(
    in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
) -> nn.Module:
    if kernel_size != 1:
        padding = 1
    else:
        padding = 0
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False,
    )


class ConvBnReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.05,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn_momentum: float = 0.05, stride: int = 1):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.dropout1 = nn.Dropout(p=0.2, inplace=True)

        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ThreeDResNet(nn.Module):
    def __init__(self, in_channels: int, n_outputs: int, n_blocks: int = 4, bn_momentum: float = 0.05, n_basefilters: int = 32):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            n_basefilters,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,)
        self.bn1 = nn.BatchNorm3d(n_basefilters, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(3, stride=2)

        if n_blocks < 2:
            raise ValueError(f"n_blocks must be at least 2, but got {n_blocks}")

        blocks = [
            ("block1", ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum))
        ]
        n_filters = n_basefilters
        for i in range(n_blocks - 1):
            blocks.append(
                (f"block{i+2}", ResBlock(n_filters, 2 * n_filters, bn_momentum=bn_momentum, stride=2))
            )
            n_filters *= 2

        self.blocks = nn.Sequential(OrderedDict(blocks))
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(n_filters, n_outputs, bias=False)

#    def get_out_features(self):
#
#        return self.out_features

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.blocks(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
