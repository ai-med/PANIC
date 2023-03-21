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
import logging

import hydra
from omegaconf import DictConfig

from torchpanic.training import train


@hydra.main(config_path="configs/", config_name="train.yaml", version_base="1.2.0")
def main(config: DictConfig):
    return train(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
