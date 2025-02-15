import sys

import torch.nn as nn
import torch.optim as optim
import yaml

from model.TransNet import TransNet
from model.baseModel import baseModel
from data.dataset import eegDataset
from utils import *
import numpy as np
import torch

# def reshape_data(data):
#     if len(data.shape) == 4:
#         reshaped_data = data.reshape(-1, data.shape[2], data.shape[3])
#         print(f"Original shape (4D): {data.shape}")
#         print(f"New shape (3D): {reshaped_data.shape}")
#         return reshaped_data
#     elif len(data.shape) == 2:
#         reshaped_data = data.reshape(-1)
#         print(f"Original shape (2D): {data.shape}")
#         print(f"New shape (1D): {reshaped_data.shape}")
#         return reshaped_data
#     else:
#         raise ValueError("Input data must be either 2D or 4D.")

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

def load_npy_files(data_file, labels_file):
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

def transform_labels(labels):
    transformed_labels = labels.copy()
    transformed_labels[transformed_labels == 2] = 0
    transformed_labels[transformed_labels == 5] = 1
    transformed_labels[transformed_labels == 6] = 2
    return transformed_labels

def prepare_data_for_person(person_data, person_labels):
    person_labels = transform_labels(person_labels)
    dataset = eegDataset(person_data, person_labels)

    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset

def train_model(data, labels, config):
    net_args = config['network_args']
    net = TransNet(**net_args)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])

    train_data, test_data = prepare_data_for_person(data, labels)

    model = baseModel(net, config, optimizer, loss_func)

    model.train_test(train_data, test_data)

def train_and_test_per_person(data, labels, config):
    num_people = len(data)
    for person_id in range(num_people):
        print(f"Training for person {person_id}...")
        person_data = data[person_id]
        person_labels = labels[person_id]

        train_model(person_data, person_labels, config)

def train_and_test_all(data, labels, config):
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    train_model(data, labels, config)

def get_data_path(config):
    if config['network_args']['num_classes'] == 2:
        return config['data_file_binary'], config['labels_file_binary']
    elif config['network_args']['num_classes'] == 3:
        return config['data_file_multiclass'], config['labels_file_multiclass']
    else:
        print("Invalid number of classes in configuration.")
        return None, None

if __name__ == "__main__":
    config = load_config("zcu-config.yaml")

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

