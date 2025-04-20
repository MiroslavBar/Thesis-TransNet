import argparse
import random
import sys
from typing import Optional, Callable, Dict, Any, Tuple, List
import torch.nn as nn
import torch.optim as optim
import yaml

from EEG_TransNet_main.model.TransNet import TransNet
from EEG_TransNet_main.model.baseModel import baseModel
from EEG_TransNet_main.data.dataset import eegDataset
from EEG_TransNet_main.utils import *


# Function to load CSV data
def load_csv_data(file_path):
    return np.loadtxt(file_path, delimiter=',')


def label_conversion_map(conversion_type: str) -> Optional[Callable[[int], Optional[int]]]:
    """
    Returns the appropriate label conversion function based on configuration.

    Args:
        conversion_type: The type of label conversion to apply.

    Returns:
        Optional function to convert labels or None if not found.
    """
    conversion_functions = {
        'two_class_ME': two_class_ME_labels,
        'two_class_MI': two_class_MI_labels,
        'four_class_ME': four_class_ME_labels,
        'four_class_MI': four_class_MI_labels,
        'two_class_labels': two_class_labels,
        'four_class_labels': four_class_labels
    }
    return conversion_functions.get(conversion_type)


def two_class_labels(label: int) -> Optional[int]:
    if label in [2, 3, 5, 6, 8, 9, 11, 12]: return 0  # Movement
    if label in [1, 4, 7, 10]: return 1  # Relax
    return None


def two_class_MI_labels(label: int) -> Optional[int]:
    if label in [5, 6, 11, 12]: return 0  # MI Movement
    if label in [4, 10]: return 1  # MI Relax
    return None


def two_class_ME_labels(label: int) -> Optional[int]:
    if label in [2, 3, 8, 9]: return 0  # ME Movement
    if label in [1, 7]: return 1  # ME Relax
    return None


def four_class_MI_labels(label: int) -> Optional[int]:
    if label in [5]: return 0  # MI left fist movement
    if label in [6]: return 1  # MI right first movement
    if label in [12]: return 2  # MI both feet movement
    if label in [4, 10]:
        if random.randint(1, 4) == 4:  # Reducing the amount of rest trials to balance the data
            return 3  # MI relax
    return None


def four_class_ME_labels(label: int) -> Optional[int]:
    if label in [2]: return 0  # ME left fist movement
    if label in [3]: return 1  # ME right first movement
    if label in [9]: return 2  # ME both feet movement
    if label in [1, 7]: return 3  # ME relax
    return None


def four_class_labels(label: int) -> Optional[int]:
    if label in [2, 5]: return 0  # Left first movement
    if label in [3, 6]: return 1  # Right first movement
    if label in [9, 12]: return 2  # Both feet movement
    if label in [1, 4, 7, 10]:return 3  # Relax
    return None


def get_subjects(csv_dir: str) -> List[str]:

    subject_ids = set()
    for file in os.listdir(csv_dir):
        if 'SIG' in file:
            subject_id = file.split('_')[1]
            subject_ids.add(subject_id)
    return sorted(subject_ids)


def preprocess_subject(
    csv_dir: str,
    subject_id: str,
    config: Dict[str, Any]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Preprocess data for a specific subject.

    Args:
        csv_dir: Directory containing CSV files
        subject_id: Subject identifier
        config: Configuration dictionary

    Returns:
        Tuple of preprocessed data and labels, or (None, None) if no valid data
    """
    num_samples = config['num_samples']
    conversion_type = config['label_conversion']
    convert_label = label_conversion_map(conversion_type)

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

def prepare_global_dataset(config: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Combine all subjects' data and prepare a global dataset.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of combined data and labels, or (None, None) if no valid data
    """
    csv_dir = config['csv_files']
    all_data, all_labels = [], []

    subjects = get_subjects(csv_dir)
    for subject_id in subjects:
        data, labels = preprocess_subject(csv_dir, subject_id, config)
        if data is not None and labels is not None:
            all_data.append(data)
            all_labels.append(labels)

    if not all_data:
        print("No valid data found for global evaluation.")
        return None, None

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_data, all_labels




def prepare_dataset(
    data: np.ndarray,
    labels: np.ndarray
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    dataset = eegDataset(data, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])

def train_per_person(model: baseModel, config: Dict[str, Any]) -> None:
    """
    Train and evaluate model for each subject individually.

    Args:
       model: Model to train
       config: Configuration dictionary
    """
    CSV_DIR = config['csv_files']
    subjects = get_subjects(CSV_DIR)
    for subject_id in subjects:
        print(f"\nProcessing subject {subject_id}...\n")

        data, labels = preprocess_subject(CSV_DIR, subject_id, config)
        if data is None or labels is None:
            continue

        train_data, test_data = prepare_dataset(data, labels)
        model.train_test(train_data, test_data)


def train_all(model: baseModel, config: Dict[str, Any]) -> None:
    """
    Train and evaluate model on combined data from all subjects.

    Args:
        model: Model to train
        config: Configuration dictionary
    """
    print("\nProcessing global evaluation (all subjects combined)...\n")

    data, labels = prepare_global_dataset(config)
    if data is None or labels is None:
        print("No valid data found for global evaluation.")
        return

    train_data, test_data = prepare_dataset(data, labels)
    model.train_test(train_data, test_data)


def train_model(config: Dict[str, Any]) -> None:
    """
    Initialize and train model based on configuration.

    Args:
        config: Configuration dictionary
    """
    net_args = config['network_args']
    net = TransNet(**net_args)
    print('Trainable Parameters in the network:', count_parameters(net))

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), config["lr"])

    model = baseModel(net, config, optimizer, loss_func)

    if config['strategy'] == 'per_person':
        # Subject-wise evaluation
        train_per_person(model, config)

    elif config['strategy'] == 'all':
        # Global evaluation
        train_all(model, config)

    else:
        print(f"Invalid strategy: {config['strategy']}. Use 'subject' or 'global'.")

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

def main() -> None:
    parser = argparse.ArgumentParser(description='EEG Classification Model')
    parser.add_argument('--config', default='shuqfa103_config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # Configure GPU based on configuration
    config = load_config(args.config)
    train_model(config)


if __name__ == '__main__':
    main()
