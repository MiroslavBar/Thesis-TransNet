import argparse
import sys
from typing import Tuple, Optional, Dict, Any, List

import torch.nn as nn
import torch.optim as optim
import yaml

from model.TransNet import TransNet
from model.baseModel import baseModel
from data.dataset import eegDataset
import numpy as np
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Config file {config_path} not found.")
        sys.exit()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit()


def load_npy_files(data_file: str, labels_file: str) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    try:
        data = np.load(data_file, allow_pickle=True)
        print(f"Data loaded successfully from {data_file}. Shape: {data.shape}")

        labels = np.load(labels_file, allow_pickle=True)
        print(f"Labels loaded successfully from {labels_file}. Shape: {labels.shape}")

        return data, labels
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None


def transform_labels(labels: np.ndarray) -> np.ndarray:
    """
    Transform original labels to a zero-based indexing system.

    Original label mapping:
    - 2 -> 0
    - 5 -> 1
    - 6 -> 2

    Args:
        labels: Original labels array.

    Returns:
        np.ndarray: Transformed labels array.
    """
    transformed_labels = labels.copy()
    transformed_labels[transformed_labels == 2] = 0
    transformed_labels[transformed_labels == 5] = 1
    transformed_labels[transformed_labels == 6] = 2
    return transformed_labels


def prepare_data_for_person(person_data: np.ndarray, person_labels: np.ndarray) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Prepare data by transforming labels and splitting into train/test sets.

    Args:
        person_data: EEG data that will be split
        person_labels: Labels corresponding to the person's EEG data

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    person_labels = transform_labels(person_labels)
    dataset = eegDataset(person_data, person_labels)

    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def train_model(data: np.ndarray, labels: np.ndarray, config: Dict[str, Any]) -> None:
    """
    Train a TransNet model using the given data, labels, and configuration.

    Args:
        data: EEG data
        labels: Corresponding labels
        config: Dictionary containing training configuration
    """
    net_args = config['network_args']
    net = TransNet(**net_args)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])

    train_data, test_data = prepare_data_for_person(data, labels)

    model = baseModel(net, config, optimizer, loss_func)

    model.train_test(train_data, test_data)


def train_and_test_per_person(data: List[np.ndarray], labels: List[np.ndarray], config: Dict[str, Any]) -> None:
    """
    Train and test a model for each person individually.

    Args:
        data: List of EEG data arrays, one per person
        labels: List of label arrays, one per person
        config: Dictionary containing training configuration
    """
    num_people = len(data)
    for person_id in range(num_people):
        print(f"Training for person {person_id}...")
        person_data = data[person_id]
        person_labels = labels[person_id]

        train_model(person_data, person_labels, config)


def train_and_test_all(data: List[np.ndarray], labels: List[np.ndarray], config: Dict[str, Any]) -> None:
    """
    Train and test a model on data from all persons combined.

    Args:
        data: List of EEG data arrays, one per person
        labels: List of label arrays, one per person
        config: Dictionary containing training configuration
    """
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    train_model(data, labels, config)

def get_data_path(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    if config['network_args']['num_classes'] == 2:
        return config['data_file_binary'], config['labels_file_binary']
    elif config['network_args']['num_classes'] == 3:
        return config['data_file_multiclass'], config['labels_file_multiclass']
    else:
        print("Invalid number of classes in configuration.")
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description='EEG Classification Model')
    parser.add_argument('--config', default='zcu-config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # Configure GPU based on configuration
    config = load_config(args.config)

    data_file, labels_file = get_data_path(config)
    if data_file is None or labels_file is None:
        print("Invalid data paths in configuration.")
        sys.exit()

    data, labels = load_npy_files(data_file, labels_file)

    if data is not None and labels is not None:
        if config['strategy'] == "all":
            train_and_test_all(data, labels, config)
        elif config['strategy'] == "per_person":
            train_and_test_per_person(data, labels, config)
        else:
            print("Invalid training strategy in configuration.")


if __name__ == "__main__":
    main()

