import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml

from model.TransNet import TransNet
from model.baseModel import baseModel
from data.dataset import eegDataset
from utils import *


# Function to load CSV data
def load_csv_data(file_path):
    return np.loadtxt(file_path, delimiter=',')


def convert_label(label):
    # return four_class_labels(label)
    return two_class_labels(label)


def two_class_labels(label):
    # 2class movement, relax
    if label in [2, 3, 5, 6, 8, 9, 11, 12]: return 0
    if label in [1, 4, 7, 10]: return 1
    return None


def four_class_labels(label):
    # 4class right hand, left hand, both feet, relax
    if label in [2, 5]: return 0
    if label in [3, 6]: return 1
    if label in [8, 9, 11, 12]: return 2
    if label in [1, 4, 7, 10]: return 3
    return None  # Exclude other labels


def get_subjects(csv_dir):
    """Extracts unique subject IDs from filenames."""
    subject_ids = set()
    for file in os.listdir(csv_dir):
        if 'SIG' in file:
            subject_id = file.split('_')[1]  # Extracts "001" from "SUB_001_SIG_01"
            subject_ids.add(subject_id)
    return sorted(subject_ids)


def preprocess_subject(csv_dir, subject_id, num_samples=1000):
    """Preprocess and save subject data to .npy files."""
    signal_files = sorted([f for f in os.listdir(csv_dir) if f'SUB_{subject_id}_SIG' in f])
    annotation_files = sorted([f for f in os.listdir(csv_dir) if f'SUB_{subject_id}_ANN' in f])

    assert len(signal_files) == len(annotation_files), f"Mismatch for subject {subject_id}"

    all_data, all_labels = [], []

    for signal_file, annotation_file in zip(signal_files, annotation_files):
        signal_data = load_csv_data(os.path.join(csv_dir, signal_file))
        annotation_data = load_csv_data(os.path.join(csv_dir, annotation_file))

        for trial_idx in range(annotation_data.shape[0]):
            trial_label = int(annotation_data[trial_idx, 0])
            converted_label = convert_label(trial_label)
            if converted_label is None:
                continue

            start_idx = int(annotation_data[trial_idx, 3]) - 1
            end_idx = int(annotation_data[trial_idx, 4]) - 1
            trial_signal = signal_data[start_idx:end_idx, :]

            # Ensure shape consistency
            if trial_signal.shape[0] < num_samples:
                trial_signal = np.pad(trial_signal, ((0, num_samples - trial_signal.shape[0]), (0, 0)), mode='constant')
            elif trial_signal.shape[0] > num_samples:
                trial_signal = trial_signal[:num_samples, :]

            # Normalize
            mean = np.mean(trial_signal, axis=0, keepdims=True)
            std = np.std(trial_signal, axis=0, keepdims=True)
            std[std == 0] = 1
            trial_signal = ((trial_signal - mean) / std).astype(np.float32)

            # Transpose to [num_channels, num_samples]
            trial_signal = trial_signal.T

            all_data.append(trial_signal)
            all_labels.append(converted_label)

    if not all_data:  # Skip subjects with no valid trials
        return None, None

    all_data = np.array(all_data, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)

    # Save data and labels to .npy files
    return all_data, all_labels


# def load_subject(csv_dir, subject_id, output_dir, num_samples=1000):
#     """Load preprocessed data for a single subject if available; otherwise, preprocess and save it."""
#     data_file = os.path.join(output_dir, f"SUB_{subject_id}_data.npy")
#     labels_file = os.path.join(output_dir, f"SUB_{subject_id}_labels.npy")
#
#     if os.path.exists(data_file) and os.path.exists(labels_file):
#         print(f"Loading preprocessed data for subject {subject_id}...")
#         data = np.load(data_file)
#         labels = np.load(labels_file)
#     else:
#         print(f"Preprocessing and saving data for subject {subject_id}...")
#         data, labels = preprocess_subject(csv_dir, subject_id, num_samples)
#     if data is None or labels is None:
#         print(f"No valid trials found for subject {subject_id}")
#         return None, None
#
#     return data, labels

def prepare_global_dataset(csv_dir, num_samples=1000):
    """Combine all subjects' data and prepare a global DataLoader."""
    all_data, all_labels = [], []

    subjects = get_subjects(csv_dir)
    for subject_id in subjects:
        data, labels = preprocess_subject(csv_dir, subject_id, num_samples)
        if data is not None and labels is not None:
            all_data.append(data)
            all_labels.append(labels)

    if not all_data:
        print("No valid data found for global evaluation.")
        return None, None

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_data, all_labels




def prepare_dataset(data, labels):
    """Prepare DataLoader from data and labels."""
    dataset = eegDataset(data, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])

def train_per_person(CSV_DIR, model):
    # Subject-wise evaluation
    subjects = get_subjects(CSV_DIR)
    for subject_id in subjects:
        print(f"\nProcessing subject {subject_id}...\n")

        data, labels = preprocess_subject(CSV_DIR, subject_id, num_samples=config['network_args']['num_samples'])

        if data is None or labels is None:
            continue

        train_data, test_data = prepare_dataset(data, labels)
        model.train_test(train_data, test_data)

def train_all(CSV_DIR, model):
    # Global evaluation
    print("\nProcessing global evaluation (all subjects combined)...\n")
    data, labels = prepare_global_dataset(CSV_DIR,
                                          num_samples=config['network_args']['num_samples'])
    if data is None or labels is None:
        print("No valid data found for global evaluation.")
        return

    train_data, test_data = prepare_dataset(data, labels)
    model.train_test(train_data, test_data)


def train_model(config):

    CSV_DIR = config["csv_files"]

    net_args = config['network_args']
    net = TransNet(**net_args)
    print('Trainable Parameters in the network:', count_parameters(net))

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), config["lr"])

    model = baseModel(net, config, optimizer, loss_func)

    if config['strategy'] == 'per_person':
        # Subject-wise evaluation
        train_per_person(CSV_DIR, model)

    elif config['strategy'] == 'all':
        # Global evaluation
        train_all(CSV_DIR, model)

    else:
        print(f"Invalid strategy: {config['strategy']}. Use 'subject' or 'global'.")

def load_config(config_path):
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


if __name__ == '__main__':
    config = load_config("103-config.yaml")
    train_model(config)