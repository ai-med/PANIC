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
import enum
from typing import Dict, Tuple

from torch import Tensor
import torchio as tio


class ModalityType(enum.IntFlag):
    MRI = 1
    PET = 2
    TABULAR = 4


DataPointType = Dict[ModalityType, Tensor]
AugmentationType = Dict[ModalityType, tio.Compose]
BatchWithLabelType = Tuple[DataPointType, DataPointType, AugmentationType, Tensor]
