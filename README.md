# EEG Classification Experiments

This repository contains code for training and evaluating a TransNet for EEG signal classification. The model is designed to classify different types of motor execution (ME) and motor imagery (MI) tasks from EEG data.

## Prerequisites

### Installation

The repository includes a `requirements.txt` file with all necessary dependencies. To install:

```bash
pip install -r requirements.txt
```

## Datasets

The project contains two packages each of which contains script for training and testing using a respective dataset. In each package there is a .yaml configuration file.
For the Brunner-9 dataset, the original implementation  has been preserved.

## data availability
- Shuqfa-103: https://data.mendeley.com/datasets/dpmtgrn8d8/4
- Brunner-9: https://www.bbci.de/competition/iv/
- Kodera-29: https://zenodo.org/records/7893847


## Running the Experiments

To run the experiment with the default configuration:

```bash
python shuqfa103_train.py
```

```bash
python kodera29_train.py
```


To specify a different configuration file:

```bash
python [dataset]_train.py --config your_custom_config.yaml
```

