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
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .tabular import NamInspector, FunctionPlotter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("-o", "--output", type=Path)

    args = parser.parse_args()

    inspector = NamInspector(args.checkpoint)
    inspector.load()

    data, samples, outputs = inspector.get_outputs()

    weights_cat_linear = inspector.get_linear_weights(revert_standardization=True)

    plotter = FunctionPlotter(
        class_names=["CN", "MCI", "AD"],
        log_scale=["Tau", "p-Tau"],
    )

    out_file = args.output
    if out_file is None:
        out_file = args.checkpoint.with_name('nam_functions.pdf')

    fig = plotter.plot(data, samples, outputs, weights_cat_linear.drop("bias"))
    fig.savefig(out_file, bbox_inches="tight", transparent=True)
    plt.close()


if __name__ == '__main__':
    main()
