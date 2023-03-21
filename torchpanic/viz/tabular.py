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
from typing import Sequence, Union

import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset

from ..datamodule.adni import AdniDataset
from ..datamodule.credit_card_fraud import PandasDataSet
from ..models.panic import PANIC
from ..models.nam import BaseNAM, NAM
from ..modules.utils import load_model_and_data
from ..datamodule.modalities import ModalityType
from .utils import map_tabular_names


def collect_tabular_data(dataset: Dataset) -> pd.DataFrame:
    data_values = []
    data_miss = []
    for i in range(len(dataset)):
        x = dataset[i][0]
        x_tab, x_miss = x[ModalityType.TABULAR]

        if isinstance(x_tab, torch.Tensor):
            x_tab = x_tab.detach().cpu().numpy()  # shape = (bs, num_features)
        if isinstance(x_miss, torch.Tensor):
            x_miss = x_miss.detach().cpu().numpy()  # shape = (bs, num_features)

        data_values.append(x_tab)
        data_miss.append(x_miss)

    if isinstance(dataset, AdniDataset):
        index = dataset.rid
        column_names = map_tabular_names(dataset.column_names)
    elif isinstance(dataset, PandasDataSet):
        index, column_names = None, None
    else:
        raise ValueError(f"Dataset of type {type(dataset)} not implemented")

    data = np.ma.array(
        np.stack(data_values, axis=0),
        mask=np.stack(data_miss, axis=0),
        copy=False,
    )
    data = pd.DataFrame(data, index=index, columns=column_names)

    return data


def create_sample_data(tabular_data: pd.DataFrame, n_samples: int = 500) -> torch.Tensor:
    samples = torch.empty((n_samples, tabular_data.shape[1]))
    for j, (_, series) in enumerate(tabular_data.iteritems()):
        if hasattr(series, "cat"):
            uniq_values = torch.from_numpy(series.cat.categories.to_numpy())
            values = torch.cat((
                uniq_values,
                # fill by repeating the last category
                torch.full((n_samples - len(uniq_values),), uniq_values[-1])
            ))
        else:
            q = series.quantile([0.01, 0.99])
            values = torch.linspace(q.iloc[0], q.iloc[1], n_samples)
        samples[:, j] = values
    return samples


def get_modality_shape(dataloader: DataLoader, modality: ModalityType):
    dataset: Union[AdniDataset, PandasDataSet] = dataloader.dataset
    return dataset[0][0][modality].shape


def iter_datasets(datamodule):
    for path in (
        datamodule.train_data, datamodule.valid_data, datamodule.test_data,
    ):
        yield AdniDataset(path, is_training=False, modalities=datamodule.modalities)


class NamInspector:
    def __init__(
        self, checkpoint_path: str, dataset_name: str = "test", device=torch.device("cuda"),
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.dataset_name = dataset_name
        self.device = device

    def load(self):
        plmodule, datamod, config = load_model_and_data(self.checkpoint_path)
        datamod.setup("fit" if self.dataset_name != "test" else self.dataset_name)

        data = pd.concat([collect_tabular_data(ds) for ds in iter_datasets(datamod)])
        idx_cat = config.model.net.nam.idx_cat_features
        for j in idx_cat:
            data.iloc[:, j] = data.iloc[:, j].astype("category")

        self._data = data
        self._model = plmodule.net.eval().to(self.device)
        self._config = config
        self._datamod = datamod

    def _get_dataloader(self):
        dsnames = {
            "fit": self._datamod.push_dataloader,
            "val": self._datamod.val_dataloader,
            "test": self._datamod.test_dataloader,
        }
        return dsnames[self.dataset_name]()

    @torch.no_grad()
    def _collect_outputs(
        self, samples_loader: DataLoader, include_missing: bool,
    ) -> Sequence[np.ndarray]:
        dataloader = self._get_dataloader()

        non_missing = torch.zeros((1, 1, self._data.shape[1]), device=self.device)
        img_data = torch.zeros(
            get_modality_shape(dataloader, ModalityType.PET), device=self.device
        ).unsqueeze(0)

        n_prototypes = self._config.model.net.protonet.n_prototypes_per_class

        model: PANIC = self._model
        outputs = []
        for x in samples_loader:
            x = x.to(self.device).unsqueeze(1)
            tab_in = torch.cat((x, non_missing.expand_as(x)), axis=1)
            img_in = img_data.expand(x.shape[0], -1, -1, -1, -1)

            logits, similarities, occurrences, nam_features_without_dropout = model(img_in, tab_in)
            # only collect outputs referring to tabular features
            feature_outputs = nam_features_without_dropout[:, n_prototypes:]

            outputs.append(feature_outputs.detach().cpu().numpy())

        if include_missing:
            tab_in = torch.ones_like(tab_in)[:1]
            img_in = img_in[:1]
            logits, similarities, occurrences, nam_features_without_dropout = model(
                img_in, tab_in
            )
            feature_outputs = nam_features_without_dropout[:, n_prototypes:]
            outputs.append(feature_outputs.detach().cpu().numpy())

        return outputs

    def _apply_inverse_transform(self, data):
        dataloader = self._get_dataloader()

        # invert standardize transform to restore original feature distributions
        dataset: AdniDataset = dataloader.dataset
        if isinstance(data, pd.DataFrame):
            data_original = pd.DataFrame(
                dataset.tabular_inverse_transform(data.values),
                index=data.index, columns=data.columns,
            )
            for col in self._data.select_dtypes(include=["category"]).columns:
                vals = data_original.loc[:, col].apply(np.rint)  # round to nearest integer
                data_original.loc[:, col] = vals.astype("category")
        else:
            data_original = dataset.tabular_inverse_transform(data)
        return data_original

    def get_outputs(self, plt_embeddings=False):
        samples = create_sample_data(self._data)
        samples_loader = DataLoader(samples, batch_size=self._config.datamodule.batch_size)

        outputs = self._collect_outputs(samples_loader, include_missing=plt_embeddings)

        samples = np.asfortranarray(samples)
        data_original = self._apply_inverse_transform(self._data)
        samples_original = self._apply_inverse_transform(samples)

        if plt_embeddings:
            samples_original = np.row_stack(
                (samples_original, 99 * np.ones(samples_original.shape[1]))
            )
            data_original = data_original.append(pd.Series(
                99, index=data_original.columns, name="MISSING"
            ))

        outputs = np.concatenate(outputs, axis=0)
        outputs = np.asfortranarray(outputs)

        # reorder columns so they are in the same order as the output of the model
        nam_config = self._config.model.net.nam
        idx = nam_config["idx_real_features"] + nam_config["idx_cat_features"]
        data_original = data_original.iloc[:, idx]
        samples_original = samples_original[:, idx]

        return data_original, samples_original, outputs

    def get_linear_weights(self, revert_standardization: bool) -> pd.DataFrame:
        nam_model: BaseNAM = self._model.nam
        bias = nam_model.bias.detach().cpu().numpy()
        weights = nam_model.cat_linear.detach().cpu().numpy()

        dataloader = self._get_dataloader()
        dataset: AdniDataset = dataloader.dataset
        # reorder columns so they are in the same order as the output of the model
        nam_config = self._config.model.net.nam
        cat_idx = nam_config["idx_cat_features"]
        columns = ["bias"] + dataset.column_names[cat_idx].tolist()

        if revert_standardization:
            mean = dataset.tabular_mean[cat_idx]
            std = dataset.tabular_stddev[cat_idx]

            weights_new = np.empty_like(weights)
            bias_new = np.empty_like(bias)
            for k in range(weights.shape[1]):
                weights_new[:, k] = weights[:, k] / std
                bias_new[:, k] = bias[:, k] - np.dot(weights_new[:, k], mean)
        else:
            bias_new = bias
            weights_new = weights

        coef = pd.DataFrame(
            np.row_stack((bias_new, weights_new)), index=map_tabular_names(columns),
        )
        coef.index.name = "feature"
        coef.columns.name = "target"
        return coef


def get_fraudnet_outputs_from_checkpoint(checkpoint_path, device=torch.device("cuda")):
    plmodule, data, config = load_model_and_data(checkpoint_path)
    data.setup("fit")

    data = collect_tabular_data(data.train_dataloader())
    samples = create_sample_data(data)
    samples_loader = DataLoader(samples, batch_size=config.datamodule.batch_size)
    non_missing = torch.zeros((1, 1, data.shape[1]), device=device)

    model: NAM = plmodule.net.eval().to(device)
    outputs = []

    with torch.no_grad():
        for x in samples_loader:
            x = x.to(device).unsqueeze(1)
            x = torch.cat((x, non_missing.expand_as(x)), axis=1)
            logits, nam_features = model.base_forward(x)
            outputs.append(nam_features.detach().cpu().numpy())
    outputs = np.concatenate(outputs)

    samples = np.asfortranarray(samples)
    outputs = np.asfortranarray(outputs)

    return data, samples, outputs


class FunctionPlotter:
    def __init__(
        self,
        class_names: Sequence[str],
        log_scale: Sequence[str],
        n_cols: int = 6,
        size: float = 2.5,
    ) -> None:
        self.class_names = class_names
        self.log_scale = frozenset(log_scale)
        self.n_cols = n_cols
        self.size = size

    def plot(
        self,
        data: pd.DataFrame,
        samples: np.ndarray,
        outputs: np.ndarray,
        categorial_coefficients: pd.Series,
    ):
        assert data.shape[1] == samples.shape[1]
        assert data.shape[1] == outputs.shape[1]
        assert samples.shape[0] == outputs.shape[0]
        assert samples.ndim == 2
        assert outputs.ndim == 3
        assert len(self.class_names) == outputs.shape[2]

        assert len(categorial_coefficients.index.difference(data.columns)) == 0

        rc_params = {
            "axes.titlesize": "small",
            "xtick.labelsize": "x-small",
            "ytick.labelsize": "x-small",
            "lines.linewidth": 2,
        }

        with mpl.rc_context(rc_params):
            return self._plot_functions(data, samples, outputs, categorial_coefficients)

    def _plot_functions(
        self,
        data: pd.DataFrame,
        samples: np.ndarray,
        outputs: np.ndarray,
        categorial_coefficients: pd.Series,
    ):
        categorical_columns = frozenset(categorial_coefficients.index)

        n_cols = self.n_cols
        n_features = data.shape[1]
        n_rows = int(np.ceil(n_features / n_cols))

        fig = plt.figure(
            figsize=(n_cols * self.size, n_rows * self.size)
        )
        gs_outer = gridspec.GridSpec(
            n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.3,
        )

        n_odds_ratios = outputs.shape[2] - 1
        palette = sns.color_palette("Set2", n_colors=n_odds_ratios)

        ref_class_name = self.class_names[0]
        legend_data = {
            f"{name} vs {ref_class_name}": c
            for name, c in zip(self.class_names[1:], palette)
        }

        ax_legend = 0
        h_ratios = [5, 1]
        for idx, (name, values) in enumerate(data.iteritems()):
            i = idx // n_cols
            j = idx % n_cols
            gs = gs_outer[i, j].subgridspec(2, 1, height_ratios=h_ratios, hspace=0.1)
            ax_top = fig.add_subplot(gs[0, 0])
            ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
            if idx == data.shape[1] - 1:
                ax_legend = ax_top

            values = values.dropna()
            for cls_idx in range(1, n_odds_ratios + 1):
                if name in categorical_columns:
                    plot_fn = self._plot_categorical
                    coef_ref = categorial_coefficients.loc[name, 0]
                    coef_new = categorial_coefficients.loc[name, cls_idx]
                    x = data.loc[:, name].cat.categories.to_numpy()
                    y = (coef_new - coef_ref) * (x - x[0])  # log-odds ratio
                else:
                    plot_fn = self._plot_continuous
                    x = samples[:, idx]
                    mid = np.searchsorted(x, data.loc[:, name].mean())
                    log_odds_mid = outputs[mid, idx, cls_idx] - outputs[mid, idx, 0]
                    log_odds = outputs[:, idx, cls_idx] - outputs[:, idx, 0]
                    y = log_odds - log_odds_mid  # log-odds ratio

                plot_fn(
                    x=x,
                    y=y,
                    values=values if cls_idx == 1 else None,
                    ax_top=ax_top,
                    ax_bot=ax_bot,
                    color=palette[cls_idx - 1],
                    label=f"Class {cls_idx}",
                )

            ax_top.axhline(0.0, color="gray")
            ax_top.tick_params(axis="x", labelbottom=False)
            ax_top.set_title(name)
            if name in self.log_scale:
                ax_top.set_xscale("log")

            if j == 0:
                ax_top.set_ylabel("log odds ratio")

        legend_kwargs = {"loc": "center left", "bbox_to_anchor": (1.05, 0.5)}
        self._add_legend(ax_legend, legend_data, legend_kwargs)

        return fig

    def _add_legend(self, ax, palette, legend_kwargs=None):
        handles = []
        labels = []
        for name, color in palette.items():
            p = Rectangle((0, 0), 1, 1)
            p.set_facecolor(color)
            handles.append(p)
            labels.append(name)

        if legend_kwargs is None:
            legend_kwargs = {"loc": "best"}

        ax.legend(handles, labels, **legend_kwargs)

    def _plot_continuous(self, x, y, values, ax_top, ax_bot, color, label):
        ax_top.plot(x, y, marker="none", color=color, label=label, zorder=2.5)
        ax_top.grid(True)

        if values is not None:
            ax_bot.boxplot(
                values.dropna(),
                vert=False,
                widths=0.75,
                showmeans=True,
                medianprops={"color": "#ff7f00"},
                meanprops={
                    "marker": "d", "markeredgecolor": "#a65628", "markerfacecolor": "none",
                },
                flierprops={"marker": "."},
                showfliers=False,
            )
            ax_bot.yaxis.set_visible(False)

    def _plot_categorical(self, x, y, values, ax_top, ax_bot, color, label):
        ax_top.step(x, y, where="mid", color=color, label=label, zorder=2.5)
        ax_top.grid(True)

        if values is not None:
            _, counts = np.unique(values, return_counts=True)
            assert len(counts) == len(x)
            ax_bot.bar(x, height=counts / counts.sum() * 100., width=0.6, color="dimgray")
