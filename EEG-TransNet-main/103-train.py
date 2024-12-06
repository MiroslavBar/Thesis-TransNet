import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model.baseModel import baseModel
from model.TransNet import TransNet
from data.dataset import eegDataset

# Function to load CSV data
def load_csv_data(file_path):
    return np.loadtxt(file_path, delimiter=',')

# Load and preprocess the data
def prepare_dataloader(csv_dir, batch_size=32, num_samples=1000, num_channels=64):
    """
    Prepares the train and test dataloaders using the modified eegDataset class.
    """
    # List all files in the directory
    signal_files = sorted([f for f in os.listdir(csv_dir) if 'SIG' in f])
    annotation_files = sorted([f for f in os.listdir(csv_dir) if 'ANN' in f])

    assert len(signal_files) == len(annotation_files), "Mismatch between signal and annotation files."

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
            trial_signal = signal_data[:, start_idx:end_idx]

            # Resample or pad trial_signal to match num_samples
            if trial_signal.shape[1] < num_samples:
                trial_signal = np.pad(trial_signal, ((0, 0), (0, num_samples - trial_signal.shape[1])), mode='constant')
            elif trial_signal.shape[1] > num_samples:
                trial_signal = trial_signal[:, :num_samples]

            # Normalize the signal
            mean = np.mean(trial_signal, axis=-1, keepdims=True)
            std = np.std(trial_signal, axis=-1, keepdims=True)
            std[std == 0] = 1  # Avoid division by zero
            trial_signal = (trial_signal - mean) / std

            # Add to dataset
            all_data.append(trial_signal)
            all_labels.append(trial_label)

    # Convert lists to numpy arrays
    all_data = np.array(all_data, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.long)

    # Create the dataset using the eegDataset class
    dataset = eegDataset(all_data, all_labels)

    # Split into training and test sets (80-20 split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader

# Main function to train the model
def train_model():
    # Define your model configuration
    config = {
        'batch_size': 64,
        'epochs': 50,
        'preferred_device': 'gpu',
        'num_classes': 4,
        'num_channels': 64,
        'sampling_rate': 160,
        'num_samples': 1600,
        'num_segs': 5,
        'nGPU': 1,
    }
    CSV_DIR = '103_data/eegmmidb'

    # Instantiate the TransNet model
    net = TransNet(num_classes=config['num_classes'])

    # Define optimizer, loss function, and scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()

    # Prepare train and test data loaders
    train_loader, test_loader = prepare_dataloader(CSV_DIR, batch_size=config['batch_size'],
                                                   num_samples=config['num_samples'])

    # Create a baseModel instance
    model = baseModel(net, config, optimizer, loss_func)

    # Train and evaluate the model
    model.train_test(train_loader, test_loader)

if __name__ == '__main__':
    train_model()
