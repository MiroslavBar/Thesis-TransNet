import argparse
import sys
from typing import Tuple, Optional, Dict, Any, List

import mne
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
import torch

from data_loading import file_manager
from data_loading.EpochEvent import EpochEvent
from data_loading.MovementType import MovementType
from data_loading.loading_utils import find_min_sampling_frequency, get_epochs, drop_half_resting, \
    transform_data_representation

from EEG_TransNet_main.model.TransNet import TransNet
from EEG_TransNet_main.model.baseModel import baseModel
from EEG_TransNet_main.data.dataset import eegDataset




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



def load_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the input data, either the already saved and preprocessed data if the file exits
    and config.save_load_preprocessed data has been set to true, otherwise reads the raw data signals from data folder.

    Each raw data are grouped together by person that the data belongs to. The raw signals are then preprocessed
    to the desired data representation set in config.data_representation.

    The shape of the returned array data array is:
        time_series: n_people, n_samples, n_channels, n_times

    The shape of the returned labels array is:
        n_people, n_samples

    :return: a tuple of 2 elements, where the first element is the preprocessed data
     and the second is labels for each data sample
    """
    data = []
    labels = []

    preprocessed_data = file_manager.load_preprocessed_data(config)
    if preprocessed_data[0] is not None:
        return preprocessed_data

    files_per_person = file_manager.group_input_files_per_person(config)

    sampling_frequency = find_min_sampling_frequency(files_per_person)
    personal_epochs = []
    for i, person_files in enumerate(files_per_person):
        if config['network_args']['num_classes'] == 3:
            left = [left for left in person_files if left.movement_type.name is MovementType.LEFT.name]
            right = [right for right in person_files if right.movement_type.name is MovementType.RIGHT.name]

            left_epochs, left_labels = get_epochs(left, MovementType.LEFT.get_epoch_event(), sampling_frequency)
            right_epochs, right_labels = get_epochs(right, MovementType.RIGHT.get_epoch_event(), sampling_frequency)

            if left_epochs is None or right_epochs is None:
                continue

            # Dropping half of the epochs representing the resting state of the patient from each set, in order to
            # try to maintain a balanced overall dataset where 1/3 is resting 1/3 is left movement and 1/3 is right
            # movement, otherwise the resting state would be much larger than the movements
            drop_half_resting(left_epochs)
            left_labels = left_epochs.events[:, 2]
            drop_half_resting(right_epochs)
            right_labels = right_epochs.events[:, 2]

            personal_epochs.append(mne.concatenate_epochs([left_epochs, right_epochs]))

            left_data = transform_data_representation(left_epochs)
            right_data = transform_data_representation(right_epochs)

            data.append(np.concatenate((left_data, right_data)))
            labels.append(np.concatenate((left_labels, right_labels)))

        elif config['network_args']['num_classes'] == 2:
            epochs, epochs_labels = get_epochs(person_files, EpochEvent.MOVEMENT_START, sampling_frequency)

            if epochs is None:
                continue

            personal_epochs.append(epochs)

            epochs_data = transform_data_representation(epochs)
            data.append(epochs_data)
            labels.append(epochs_labels)

    data = np.array(data, dtype=object)
    labels = np.array(labels, dtype=object)


    file_manager.save_preprocessed_data(data, labels, config)

    return data, labels


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
    parser.add_argument('--config', default='kodera29_config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # Configure GPU based on configuration
    config = load_config(args.config)

    # Load data
    data, labels = load_data(config)

    if data is not None and labels is not None:
        if config['strategy'] == "all":
            train_and_test_all(data, labels, config)
        elif config['strategy'] == "per_person":
            train_and_test_per_person(data, labels, config)
        else:
            print("Invalid training strategy in configuration.")


if __name__ == "__main__":
    main()

