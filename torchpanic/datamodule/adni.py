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
import collections.abc
import copy
import logging
from operator import itemgetter
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import torchio as tio

from .modalities import AugmentationType, DataPointType, ModalityType

LOG = logging.getLogger(__name__)

DIAGNOSIS_MAP = {"CN": 0, "MCI": 1, "Dementia": 2}
DIAGNOSIS_MAP_BINARY = {"CN": 0, "Dementia": 1}


def Identity(x):
    return x


def get_image_transform(p=0.0, rotate=0, translate=0, scale=0):
    # Image sizes  PET & MRI        Dataset {113, 137, 113}
    img_transforms = []

    randomAffineWithRot = tio.RandomAffine(
        scales=scale,
        degrees=rotate,
        translation=translate,
        image_interpolation="linear",
        default_pad_value="otsu",
        p=p,  # no transform if validation
    )
    img_transforms.append(randomAffineWithRot)

    img_transform = tio.Compose(img_transforms)
    return img_transform


class AdniDataset(Dataset):
    def __init__(
        self,
        path: str,
        modalities: Union[ModalityType, int, Sequence[str]],
        augmentation: Dict[str, Any],
    ) -> None:
        self.path = path
        if isinstance(modalities, collections.abc.Sequence):
            mod_type = ModalityType(0)
            for mod in modalities:
                mod_type |= getattr(ModalityType, mod)
            self.modalities = mod_type
        else:
            self.modalities = ModalityType(modalities)
        self.augmentation = augmentation

        self._set_transforms(augmentations=self.augmentation)
        self._load()

    def _set_transforms(self, augmentations: Dict) -> None:
        self.transforms = {}
        if ModalityType.MRI in self.modalities:
            self.transforms[ModalityType.MRI] = get_image_transform(**augmentations)
        if ModalityType.PET in self.modalities:
            self.transforms[ModalityType.PET] = get_image_transform(**augmentations)
        if ModalityType.TABULAR in self.modalities:
            self.transforms[ModalityType.TABULAR] = Identity

    def _load(self) -> None:
        data_points = {
            flag: [] for flag in ModalityType.__members__.values() if flag in self.modalities
        }
        load_mri = ModalityType.MRI in self.modalities
        load_pet = ModalityType.PET in self.modalities
        load_tab = ModalityType.TABULAR in self.modalities

        LOG.info("Loading %s from %s", self.modalities, self.path)

        diagnosis = []
        rid = []
        column_names = None
        with h5py.File(self.path, mode='r') as file:
            if load_tab:
                tab_stats = file['stats/tabular']
                tab_mean = tab_stats['mean'][:]
                tab_std = tab_stats['stddev'][:]
                assert np.all(tab_std > 0), "stddev is not positive"
                column_names = tab_stats['columns'][:]
                self._tab_mean = tab_mean
                self._tab_std = tab_std

            for name, group in file.items():
                if name == "stats":
                    continue

                data_point = []
                if load_mri:
                    mri_data = group['MRI/T1/data'][:]
                    data_point.append(
                        tio.Subject(
                            image=tio.ScalarImage(tensor=mri_data[np.newaxis])
                        )
                    )

                if load_pet:
                    pet_data = group['PET/FDG/data'][:]
                    # pet_data = np.nan_to_num(pet_data, copy=False)
                    data_point.append(
                        tio.Subject(
                            image=tio.ScalarImage(tensor=pet_data[np.newaxis])
                        )
                    )

                if load_tab:
                    tab_values = group['tabular/data'][:]
                    tab_missing = group['tabular/missing'][:]
                    # XXX: always assumes that mean and std are from the training data
                    tab_data = np.stack((
                        (tab_values - tab_mean) / tab_std,
                        tab_missing,
                    ))
                    data_point.append(tab_data)

                assert len(data_points) == len(data_point)
                for samples, data in zip(data_points.values(), data_point):
                    samples.append(data)

                diagnosis.append(group.attrs['DX'])
                rid.append(group.attrs['RID'])

        LOG.info("Loaded %d samples", len(rid))

        dmap = DIAGNOSIS_MAP
        labels, counts = np.unique(diagnosis, return_counts=True)
        assert len(labels) == len(dmap), f"expected {len(dmap)} labels, but got {labels}"
        LOG.info("Classes: %s", pd.Series(counts, index=labels))

        self._column_names = column_names
        self._data_points = data_points
        self._diagnosis = [dmap[d] for d in diagnosis]
        self._rid = rid

    @property
    def rid(self):
        return self._rid

    @property
    def column_names(self):
        return self._column_names

    @property
    def tabular_mean(self):
        return self._tab_mean

    @property
    def tabular_stddev(self):
        return self._tab_std

    def tabular_inverse_transform(self, values):
        values_arr = np.ma.atleast_2d(values)
        if len(self._tab_mean) != values_arr.shape[1]:
            raise ValueError(f"expected {len(self._tab_mean)} features, but got {values_arr.shape[1]}")

        vals_t = values_arr * self._tab_std[np.newaxis] + self._tab_mean[np.newaxis]
        return vals_t.reshape(values.shape)

    def get_tabular(self, index: int, inverse_transform: bool = False) -> np.ma.array:
        tab_vals, tab_miss = self[index][0][ModalityType.TABULAR]
        tab_vals = np.ma.array(tab_vals, mask=tab_miss)
        if inverse_transform:
            return self.tabular_inverse_transform(tab_vals)
        return tab_vals

    def __len__(self) -> int:
        return len(self._rid)

    def _as_tensor(self, x):
        if isinstance(x, tio.Subject):
            return x.image.data
        return x

    def __getitem__(self, index: int) -> Tuple[DataPointType, DataPointType, AugmentationType, int]:
        label = self._diagnosis[index]
        sample = {}
        sample_raw = {}
        augmentations = {}
        for modality_id, samples in self._data_points.items():
            data_raw = samples[index]
            data_transformed = self.transforms[modality_id](data_raw)
            if isinstance(data_transformed, tio.Subject):
                augmentations[modality_id] = data_transformed.get_composed_history()

            sample_raw[modality_id] = self._as_tensor(data_raw)
            sample[modality_id] = self._as_tensor(data_transformed)

        return sample, sample_raw, augmentations, label


def collate_adni(batch):
    get = itemgetter(0, 1, 3)  # 2nd position is a tio.Transform instance
    batch_wo_aug = [get(elem) for elem in batch]

    keys = batch[0][2].keys()
    augmentations = {k: [elem[2][k] for elem in batch] for k in keys}

    batch_stacked = default_collate(batch_wo_aug)
    return batch_stacked[:-1] + [augmentations] + batch_stacked[-1:]


class AdniDataModule(pl.LightningDataModule):
    def __init__(
        self,
        modalities: Union[ModalityType, int],
        train_data: str,
        valid_data: str,
        test_data: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        augmentation: Dict[str, Any] = {"p": 0.0},
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.modalities = modalities
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        self.augmentation = augmentation

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = AdniDataset(self.train_data, modalities=self.modalities, augmentation=self.augmentation)
            self.push_dataset = copy.deepcopy(self.train_dataset)
            self.push_dataset._set_transforms(augmentations={"p": 0})
            self.eval_dataset = AdniDataset(self.valid_data, modalities=self.modalities, augmentation={"p": 0})
        elif stage == 'test' and self.test_data is not None:
            self.test_dataset = AdniDataset(self.test_data, modalities=self.modalities, augmentation={"p": 0})
            self.eval_dataset = AdniDataset(self.valid_data, modalities=self.modalities, augmentation={"p": 0})

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
            collate_fn=collate_adni,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers // 2,
            pin_memory=True,
            collate_fn=collate_adni,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_adni,
        )

    def push_dataloader(self):
        return DataLoader(
            self.push_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_adni,
        )
