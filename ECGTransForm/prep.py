import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_ecg5000(pickle_file, output_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    labels = data[:, 0]  # First column is the label
    samples = data[:, 1:]  # Remaining columns are features

    samples = samples.reshape(samples.shape[0], 1, samples.shape[1])

    dataset = {
        "samples": torch.tensor(samples, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

    torch.save(dataset, output_file)

def split_validation_data(validation_pickle, val_output, test_output, test_size=0.2, random_state=42):
    with open(validation_pickle, 'rb') as f:
        data = pickle.load(f)

    labels = data[:, 0]  # First column is the label
    samples = data[:, 1:]  # Remaining columns are features

    samples_train, samples_test, labels_train, labels_test = train_test_split(
        samples, labels, test_size=test_size, random_state=random_state
    )

    val_dataset = {
        "samples": torch.tensor(samples_train, dtype=torch.float32),
        "labels": torch.tensor(labels_train, dtype=torch.long)
    }
    torch.save(val_dataset, val_output)

    test_dataset = {
        "samples": torch.tensor(samples_test, dtype=torch.float32),
        "labels": torch.tensor(labels_test, dtype=torch.long)
    }
    torch.save(test_dataset, test_output)

preprocess_ecg5000("../ECG5000_train.pickle", "data/ecg5000/train.pt")

split_validation_data(
    "../ECG5000_validation.pickle",
    "data/ecg5000/val.pt",
    "data/ecg5000/test.pt"
)
