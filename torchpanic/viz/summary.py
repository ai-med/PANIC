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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

def summary_plot(
    predictions,
    feature_names=None,
    max_display=20,
    title="",
    class_names=None,
    class_inds=None,
    colormap="Set2",
    axis_color="black",
    sort=True,
    plot_size="auto",
):
    """Create a summary plot of shapley values for 1-dimensional features, colored by feature values when they are provided.
    This method is based on Slundberg - SHAP (https://github.com/slundberg/shap/blob/master/shap/)
    Args:
        predictions : list of numpy.array
            For each class, a list of predictions for each function with shape = (n_samples, n_features).
        feature_names : list
            Names of the features (length # features)
        max_display : int
            How many top features to include in the plot (default is 20)
        plot_size : "auto" (default), float, (float, float), or None
            What size to make the plot. By default the size is auto-scaled based on the number of
            features that are being displayed. Passing a single float will cause each row to be that
            many inches high. Passing a pair of floats will scale the plot by that
            number of inches. If None is passed then the size of the current figure will be left
            unchanged.
    """

    multi_class = True

    # default color:
    if multi_class:
        #cm = mpl.colormaps[colormap]
        cm = sns.color_palette(colormap, n_colors=3)
        #color = lambda i: cm(i)
        color = lambda i: cm[(i + 2) % 3]

    num_features = predictions[0].shape[1] if multi_class else predictions.shape[1]

    shape_msg = (
        "The shape of the shap_values matrix does not match the shape of the "
        "provided data matrix."
    )
    if num_features - 1 == len(feature_names):
        raise ValueError(
            shape_msg
            + " Perhaps the extra column in the shap_values matrix is the "
            "constant offset? If so just pass shap_values[:,:-1]."
        )
    elif num_features != len(feature_names):
        raise ValueError(shape_msg)

    if sort:
        # order features by the sum of their effect magnitudes
        if multi_class:
            feature_order = np.argsort(
                np.sum(np.mean(np.abs(predictions), axis=0), axis=0)
            )
        else:
            feature_order = np.argsort(np.sum(np.abs(predictions), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)) :]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    row_height = 0.4
    if plot_size == "auto":
        figsize = (8, len(feature_order) * row_height + 1.5)
    elif isinstance(plot_size, (list, tuple)):
        figsize = (plot_size[0], plot_size[1])
    elif plot_size is not None:
        figsize = (8, len(feature_order) * plot_size + 1.5)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(x=0, color="#999999", zorder=-1)

    legend_handles = []
    legend_text = []

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(predictions))]
    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds))
    left_pos = np.zeros(len(feature_inds))

    if class_inds is None:
        class_inds = np.argsort([-np.abs(p).mean() for p in predictions])
    elif class_inds == "original":
        class_inds = range(len(predictions))
    for i, ind in enumerate(class_inds):
        global_importance = np.abs(predictions[ind]).mean(0)
        ax.barh(
            y_pos, global_importance[feature_inds], 0.7, left=left_pos, align='center',
            color=color(i), label=class_names[ind]
        )
        left_pos += global_importance[feature_inds]
    f_names_relevant = [feature_names[i] for i in feature_inds]
    ax.set_yticklabels(f_names_relevant)

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)

    plt.yticks(
        range(len(feature_order)),
        [feature_names[i] for i in feature_order],
        fontsize='large'
    )
    plt.ylim(-1, len(feature_order))

    ax.set_title(title)
    ax.tick_params(color=axis_color, labelcolor=axis_color)
    ax.tick_params("y", length=20, width=0.5, which="major")
    ax.tick_params("x")  # , labelsize=11)
    ax.xaxis.grid(True)
    ax.set_xlabel("Mean Importance", fontsize='large')

    # legend_handles.append(missing_handle)
    # legend_text.append("Missing")
    if len(legend_handles) > 0:
        plt.legend(
            legend_handles[::-1],
            legend_text[::-1],
            loc="center right",
            bbox_to_anchor=(1.30, 0.55),
            frameon=False,
    )
    else:
        plt.legend(
            loc="best",
            frameon=True,
            fancybox=True,
            facecolor='white',
            framealpha=1.0,
            fontsize=12,
        )

    return fig, f_names_relevant
