# Don't PANIC: Prototypical Additive Neural Network for Interpretable Classification of Alzheimer's Disease

[![Conference Paper](https://img.shields.io/static/v1?label=DOI&message=10.1007%2f978-3-031-34048-2_7&color=3a7ebb)](https://dx.doi.org/10.1007/978-3-031-34048-2_7)
[![Preprint](https://img.shields.io/badge/arXiv-2303.07125-b31b1b)](https://arxiv.org/abs/2303.07125)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

This repository contains the code to the paper "Don't PANIC: Prototypical Additive Neural Network for Interpretable Classification of Alzheimer's Disease"

```
@inproceedings{Wolf2023-IPMI,
  doi = {10.1007/978-3-031-34048-2_7},
  author = {Wolf, Tom Nuno and P{\"o}lsterl, Sebastian and Wachinger, Christian},
  title = {Don't PANIC: Prototypical Additive Neural Network for Interpretable Classification of Alzheimer's Disease},
  booktitle = {Information Processing in Medical Imaging},
  pages = {82--94},
  year = {2023}
}
```

If you are using this code, please cite the paper above.


## Installation

Use [conda](https://conda.io/miniconda.html) to create an environment called `panic` with all dependencies:

```bash
conda env create -n panic --file requirements.yaml
```

Additionally, install the package torchpanic from this repository with
```bash
pip install --no-deps -e .
```

## Data

We used data from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/).
Since we are not allowed to share our data, you would need to process the data yourself.
Data for training, validation, and testing should be stored in separate
[HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) files,
using the following hierarchical format:

1. First level: A unique identifier.
2. The second level always has the following entries:
    1. A group named `PET` with the subgroup `FDG`, which itself has the
       [dataset](https://docs.h5py.org/en/stable/high/dataset.html) named `data` as child:
       The graymatter density map of size (113,117,113). Additionally, the subgroup `FDG` has an attribute `imageuid` with is the unique image identifier.
    2. A group  named `tabular`, which has two datasets called `data` and `missing`, each of size 41:
       `data` contains the tabular data values, while `missing` is a missing value indicator if a tabular feature was not acquired at this visit.
    3. A scalar [attribute](https://docs.h5py.org/en/stable/high/attr.html) `RID` with the *patient* ID.
    4. A string attribute `VISCODE` with ADNI's visit code.
    5. A string attribute `DX` containing the diagnosis (`CN`, `MCI` or `Dementia`).

One entry in the resulting HDF5 file should have the following structure:
```
/1010012                 Group
    Attribute: RID scalar
        Type:      native long
        Data:  1234
    Attribute: VISCODE scalar
        Type:      variable-length null-terminated UTF-8 string
        Data:  "bl"
    Attribute: DX scalar
        Type:      variable-length null-terminated UTF-8 string
        Data:  "CN"
/1010012/PET Group
/1010012/PET/FDG Group
    Attribute imageuid scalar
        Type:      variable-length null-terminated UTF-8 string
        Data: "12345"
/1010012/PET/FDG/data Dataset {113, 137, 133}
/1010012/tabular Group
/1010012/tabular/data Dataset {41}
/1010012/tabular/missing Dataset {41}
```

Finally, the HDF5 file should also contain the following meta-information
in a separate group named `stats`:

```
/stats/tabular           Group
/stats/tabular/columns   Dataset {41}
/stats/tabular/mean      Dataset {41}
/stats/tabular/stddev    Dataset {41}
```

They are the names of the features in the tabular data,
their mean, and standard deviation.

## Usage

PANIC processes tabular data depending on its data type.
Therefore, it is necessary to tell PANIC how to process each tabular feature:
The following indices must be given to the model in the configs file `configs/model/panic.yaml`:

`idx_real_features`: indices of real-valued features within `tabular` data.
`idx_cat_features`: indices of categorical features within `tabular` data.
`idx_real_has_missing`: indices of real-valued features which should be considered from `missing`.
`idx_cat_has_missing`: indices of categorical features which should be considered from `missing`.

Similarly, missing tabular inputs to DAFT (`configs/model/daft.yaml`) need to be specified with `idx_tabular_has_missing`.

## Training

To train PANIC, or any of the baseline models, adapt the config files (mainly `train.yaml`) and  execute the `train.py` script to begin training.

Model checkpoints will be written to the `outputs` folder by default.


## Interpretation of results

We provide some useful utility function to create plots and visualization required to interpret the model.
You can find them under `torchpanic/viz`.

