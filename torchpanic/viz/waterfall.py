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
# Code has been adapted from https://github.com/slundberg/shap/, which
# has been released under the following license:
# The MIT License (MIT)
# 
# Copyright (c) 2018 Scott Lundberg
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
import re

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.transforms import ScaledTranslation
import numpy as np


def format_value(s, format_str):
    """ Strips trailing zeros and uses a unicode minus sign.
    """
    if np.ma.isMaskedArray(s) and np.ma.is_masked(s):
        return "missing"

    if not isinstance(s, str):
        s = format_str % s
    s = re.sub(r'\.?0+$', '', s)
    if s[0] == "-":
        s = u"\u2212" + s[1:]
    return s



class WaterfallPlotter:
    def __init__(self, n_prototypes: int) -> None:
        self.n_prototypes = n_prototypes

    def plot(
        self,
        expected_value,
        shap_values,
        features=None,
        feature_names=None,
        actual_class_name=None,
        predicted_class_name=None,
        max_display=10,
    ):
        """ Plots an explantion of a single prediction as a waterfall plot.
        The SHAP value of a feature represents the impact of the evidence provided by that feature on the model's
        output. The waterfall plot is designed to visually display how the SHAP values (evidence) of each feature
        move the model output from our prior expectation under the background data distribution, to the final model
        prediction given the evidence of all the features. Features are sorted by the magnitude of their SHAP values
        with the smallest magnitude features grouped together at the bottom of the plot when the number of features
        in the models exceeds the max_display parameter.
        Parameters
        ----------
        expected_value : float
            This is the reference value that the feature contributions start from. For SHAP values it should
            be the value of explainer.expected_value.
        shap_values : numpy.array
            One dimensional array of SHAP values.
        features : numpy.array
            One dimensional array of feature values. This provides the values of all the
            features, and should be the same shape as the shap_values argument.
        feature_names : list
            List of feature names (# features).
        actual_class_name : str
            Name of actual class.
        predicted_class_name : str
            Name of predicted class.
        max_display : str
            The maximum number of features to plot.
        """

        # init variables we use for tracking the plot locations
        num_features = min(max_display, len(shap_values))
        row_height = 0.2
        rng = range(num_features - 1, -1, -1)
        order = np.argsort(-np.abs(shap_values))
        pos_lefts = []
        pos_inds = []
        pos_widths = []
        neg_lefts = []
        neg_inds = []
        neg_widths = []
        loc = expected_value + shap_values.sum()
        yticklabels = ["" for i in range(num_features + 1)]

        # size the plot based on how many features we are plotting
        fig = plt.figure(figsize=(8, num_features * row_height + 1.5))

        # see how many individual (vs. grouped at the end) features we are plotting
        if num_features == len(shap_values):
            num_individual = num_features
        else:
            num_individual = num_features - 1

        # compute the locations of the individual features and plot the dashed connecting lines
        for i in range(num_individual):
            sval = shap_values[order[i]]
            loc -= sval
            if sval >= 0:
                pos_inds.append(rng[i])
                pos_widths.append(sval)
                pos_lefts.append(loc)
            else:
                neg_inds.append(rng[i])
                neg_widths.append(sval)
                neg_lefts.append(loc)
            if num_individual != num_features or i + 4 < num_individual:
                plt.plot(
                    [loc, loc],
                    [rng[i] - 1 - 0.4, rng[i] + 0.4],
                    color="#bbbbbb",
                    linestyle="--",
                    linewidth=0.5,
                    zorder=-1,
                )
            if features is None or order[i] < self.n_prototypes:
                yticklabels[rng[i]] = feature_names[order[i]]
            else:
                yticklabels[rng[i]] = (
                    feature_names[order[i]]
                    + " = "
                    + format_value(features[order[i]], "%0.03f")
                )

        # add a last grouped feature to represent the impact of all the features we didn't show
        if num_features < len(shap_values):
            yticklabels[0] = "%d other features" % (len(shap_values) - num_features + 1)
            remaining_impact = expected_value - loc
            if remaining_impact < 0:
                pos_inds.append(0)
                pos_widths.append(-remaining_impact)
                pos_lefts.append(loc + remaining_impact)
            else:
                neg_inds.append(0)
                neg_widths.append(-remaining_impact)
                neg_lefts.append(loc + remaining_impact)

        points = (
            pos_lefts
            + list(np.array(pos_lefts) + np.array(pos_widths))
            + neg_lefts
            + list(np.array(neg_lefts) + np.array(neg_widths))
        )
        dataw = np.max(points) - np.min(points)

        # draw invisible bars just for sizing the axes
        label_padding = np.array([0.1 * dataw if w < 1 else 0 for w in pos_widths])
        plt.barh(
            pos_inds,
            np.array(pos_widths) + label_padding + 0.02 * dataw,
            left=np.array(pos_lefts) - 0.01 * dataw,
            color=colors.to_rgba_array("r"),
            alpha=0,
        )
        label_padding = np.array([-0.1 * dataw if -w < 1 else 0 for w in neg_widths])
        plt.barh(
            neg_inds,
            np.array(neg_widths) + label_padding - 0.02 * dataw,
            left=np.array(neg_lefts) + 0.01 * dataw,
            color=colors.to_rgba_array("b"),
            alpha=0,
        )

        # define variable we need for plotting the arrows
        head_length = 0.08
        bar_width = 0.8
        xlen = plt.xlim()[1] - plt.xlim()[0]
        fig = plt.gcf()
        ax = plt.gca()
        xticks = ax.get_xticks()
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        bbox_to_xscale = xlen / width
        hl_scaled = bbox_to_xscale * head_length
        renderer = fig.canvas.get_renderer()

        # draw the positive arrows
        for i in range(len(pos_inds)):
            dist = pos_widths[i]
            arrow_obj = plt.arrow(
                pos_lefts[i],
                pos_inds[i],
                max(dist - hl_scaled, 0.000001),
                0,
                head_length=min(dist, hl_scaled),
                color="tab:red",
                width=bar_width,
                head_width=bar_width,
            )

            txt_obj = plt.text(
                pos_lefts[i] + 0.5 * dist,
                pos_inds[i],
                format_value(pos_widths[i], "%+0.02f"),
                horizontalalignment="center",
                verticalalignment="center",
                color="white",
                fontsize=12,
            )
            text_bbox = txt_obj.get_window_extent(renderer=renderer)
            arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

            # if the text overflows the arrow then draw it after the arrow
            if text_bbox.width > arrow_bbox.width:
                txt_obj.remove()

        # draw the negative arrows
        for i in range(len(neg_inds)):
            dist = neg_widths[i]

            arrow_obj = plt.arrow(
                neg_lefts[i],
                neg_inds[i],
                -max(-dist - hl_scaled, 0.000001),
                0,
                head_length=min(-dist, hl_scaled),
                color="tab:blue",
                width=bar_width,
                head_width=bar_width,
            )

            txt_obj = plt.text(
                neg_lefts[i] + 0.5 * dist,
                neg_inds[i],
                format_value(neg_widths[i], "%+0.02f"),
                horizontalalignment="center",
                verticalalignment="center",
                color="white",
                fontsize=12,
            )
            text_bbox = txt_obj.get_window_extent(renderer=renderer)
            arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

            # if the text overflows the arrow then draw it after the arrow
            if text_bbox.width > arrow_bbox.width:
                txt_obj.remove()

        # draw the y-ticks twice, once in gray and then again with just the feature names in black
        # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
        ytick_pos = np.arange(num_features)
        plt.yticks(ytick_pos, yticklabels[:-1], fontsize=13)

        # put horizontal lines for each feature row
        for i in range(num_features):
            plt.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

        # mark the prior expected value and the model prediction
        plt.axvline(
            expected_value,
            color="#bbbbbb",
            linestyle="--",
            linewidth=0.85,
            zorder=-1,
        )
        fx = expected_value + shap_values.sum()
        plt.axvline(fx, color="#bbbbbb", linestyle="--", linewidth=0.85, zorder=-1)

        # clean up the main axis
        plt.gca().xaxis.set_ticks_position("bottom")
        plt.gca().yaxis.set_ticks_position("none")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        ax.tick_params(labelsize=13)
        # plt.xlabel("Model output", fontsize=12)

        # draw the E[f(X)] tick mark
        xmin, xmax = ax.get_xlim()
        xticks = ax.get_xticks()
        xticks = list(xticks)
        min_ind = 0
        min_diff = 1e10
        for i in range(len(xticks)):
            v = abs(xticks[i] - expected_value)
            if v < min_diff:
                min_diff = v
                min_ind = i
        xticks.pop(min_ind)
        ax.set_xticks(xticks)
        ax.tick_params(labelsize=13)
        ax.set_xlim(xmin, xmax)

        ax2 = ax.twiny()
        ax2.set_xlim(xmin, xmax)
        ax2.set_xticks([expected_value, expected_value + 1e-8])
        ax2.set_xticklabels(["\nbias", "\n$ = " + format_value(expected_value, "%0.03f") + "$"], fontsize=12, ha="left")
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)

        # draw the f(x) tick mark
        ax3 = ax2.twiny()
        ax3.set_xlim(xmin, xmax)
        ax3.set_xticks([fx, fx + 1e-8])
        the_class = "^{\\mathrm{%s}}" % predicted_class_name if predicted_class_name is not None else ""
        ax3.set_xticklabels(
            [f"$\\mu{the_class}$", "$ = " + format_value(fx, "%0.03f") + "$"], fontsize=12, ha="left"
        )
        tick_labels = ax3.xaxis.get_majorticklabels()
        tick_labels[0].set_transform(
            tick_labels[0].get_transform()
            + ScaledTranslation(-10 / 72.0, 0, fig.dpi_scale_trans)
        )
        tick_labels[1].set_transform(
            tick_labels[1].get_transform()
            + ScaledTranslation(12 / 72.0, 0, fig.dpi_scale_trans)
        )
        # tick_labels[1].set_color("#999999")
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.spines["left"].set_visible(False)

        # adjust the position of the E[f(X)] = x.xx label
        tick_labels = ax2.xaxis.get_majorticklabels()
        tick_labels[0].set_transform(
            tick_labels[0].get_transform()
            + ScaledTranslation(-14 / 72.0, 0, fig.dpi_scale_trans)
        )
        tick_labels[1].set_transform(
            tick_labels[1].get_transform()
            + ScaledTranslation(
                11 / 72.0, -1 / 72.0, fig.dpi_scale_trans
            )
        )
        # tick_labels[1].set_color("#999999")

        the_title = []
        if predicted_class_name is not None:
            the_title.append(f"Predicted: {predicted_class_name}")
        if actual_class_name is not None:
            the_title.append(f"Actual: {actual_class_name}")
        if len(the_title) > 0:
            plt.title(", ".join(the_title))

        return fig
