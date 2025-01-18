import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model.baseModel import baseModel
from model.TransNet import TransNet
from data.dataset import eegDataset

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from visdom import Visdom
from model.TransNet import TransNet
from model.baseModel import baseModel
import time
import os
import yaml
from data.data_utils import *
from data.dataset import eegDataset
from utils import *
import time

# Function to load CSV data
def load_csv_data(file_path):
    return np.loadtxt(file_path, delimiter=',')

# Load and preprocess the data
def prepare_dataloader(csv_dir, batch_size=32, num_samples=1000, num_channels=64, max_files=20):
    """
    Prepares the train and test datasets (instances of eegDataset).
    """
    # List all files in the directory
    signal_files = sorted([f for f in os.listdir(csv_dir) if 'SIG' in f])
    annotation_files = sorted([f for f in os.listdir(csv_dir) if 'ANN' in f])

    assert len(signal_files) == len(annotation_files), "Mismatch between signal and annotation files."

    signal_files = signal_files[:max_files]
    annotation_files = annotation_files[:max_files]

    # Process signals and annotations to create the dataset
    all_data = []
    all_labels = []

    for signal_file, annotation_file in zip(signal_files, annotation_files):
        # Load signal and annotation data
        signal_data = load_csv_data(os.path.join(csv_dir, signal_file))
        annotation_data = load_csv_data(os.path.join(csv_dir, annotation_file))

        for trial_idx in range(annotation_data.shape[0]):
            # Extract label and trial data
            trial_label = int(annotation_data[trial_idx, 0])
            start_idx = int(annotation_data[trial_idx, 3]) - 1
            end_idx = int(annotation_data[trial_idx, 4]) - 1

            # Extract signal slice: (num_samples, num_channels)
            trial_signal = signal_data[start_idx:end_idx, :]  # Adjust for new shape

            # Resample or pad trial_signal to match num_samples
            if trial_signal.shape[0] < num_samples:
                trial_signal = np.pad(trial_signal, ((0, num_samples - trial_signal.shape[0]), (0, 0)), mode='constant')
            elif trial_signal.shape[0] > num_samples:
                trial_signal = trial_signal[:num_samples, :]

            # Normalize the signal along the sample axis
            mean = np.mean(trial_signal, axis=0, keepdims=True)
            std = np.std(trial_signal, axis=0, keepdims=True)
            std[std == 0] = 1  # Avoid division by zero
            trial_signal = ((trial_signal - mean) / std).astype(np.float32)

            # Add to dataset
            all_data.append(trial_signal)
            all_labels.append(trial_label)

    # Convert lists to numpy arrays
    all_data = np.array(all_data, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)

    # Create the dataset using the eegDataset class
    dataset = eegDataset(all_data, all_labels)

    # Split into training and test sets (80-20 split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Instead of returning DataLoaders, return the datasets themselves
    return train_dataset, test_dataset


# Main function to train the model
def train_model():
    # Define your model configuration
    config = {
         # Existing parameters
     # Existing parameters
    'batch_size': 64,
    'epochs': 50,
    'preferred_device': 'gpu',
    'sampling_rate': 160,
    'num_segs': 5,
    'nGPU': 1,

    # Parameters from the second config
    'num_classes': 4,           # Already present in both
    'num_channels': 64,         # Overwrites previous num_channels (64) with 22
    'num_samples': 1000,        # Overwrites previous num_samples (1600) with 1000
    'embed_dim': 32,
    'pool_size': 50,
    'pool_stride': 15,
    'num_heads': 8,
    'fc_ratio': 4,
    'depth': 4,
    'attn_drop': 0.5,
    'fc_drop': 0.5
    }
    CSV_DIR = '103_data/eegmmidb'

    # Instantiate the TransNet model
    # net = TransNet(num_classes=config['num_classes'])


    # Define optimizer, loss function, and scheduler
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # loss_func = torch.nn.CrossEntropyLoss()
    net_args = {
        'num_classes': config['num_classes'],
        'num_channels': config['num_channels'],
        'embed_dim': config['embed_dim'],
        'pool_size': config['pool_size'],
        'pool_stride': config['pool_stride'],
        'num_heads': config['num_heads'],
        'fc_ratio': config['fc_ratio'],
        'depth': config['depth'],
        'attn_drop': config['attn_drop'],
        'fc_drop': config['fc_drop']
    }
    net = TransNet(**net_args)
    print('Trainable Parameters in the network are: ' + str(count_parameters(net)))

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    # Prepare train and test data loaders
    train_loader, test_loader = prepare_dataloader(CSV_DIR, batch_size=config['batch_size'],
                                                   num_samples=config['num_samples'])

    # Create a baseModel instance
    model = baseModel(net, config, optimizer, loss_func)

    # Train and evaluate the model
    model.train_test(train_loader, test_loader)

if __name__ == '__main__':
    train_model()
