import torch
from torch.utils.data import Dataset
import numpy as np
from audiomentations import Compose, PitchShift
import pickle


class AugmentedDataset(Dataset):
    def __init__(self, data, labels, augment=True):
        self.data = data
        self.labels = labels
        self.augment = augment
        self.augmenter = Compose([
            #AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.1),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.001),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]

        if self.augment:
            signal = self.apply_augmentation(signal)

        return signal, label


    def apply_augmentation(self, signal):
        signal = self.augmenter(samples=signal, sample_rate=16000)
        return signal

class CustomDataset:
    def __init__(self, data, labels, name):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels
        self.name = np.array(list(name.values())) if isinstance(name, dict) else np.array(name) if not isinstance(name, np.ndarray) else name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.name is not None and len(self.name) > 0:
            return self.data[idx], self.labels[idx], self.name[idx]
        else:
            return self.data[idx], self.labels[idx],"none"


    def print_shapes(self):
        print(f"data : {self.data.shape}, labels: {self.labels.shape}, names : {self.name.shape}")

    def pop(self, idx):
        data_item = self.data[idx]
        label_item = self.labels[idx]
        name_item = self.name[idx]
        
        self.data = np.delete(self.data, idx, axis=0)
        self.labels = np.delete(self.labels, idx, axis=0)
        self.name = np.delete(self.name, idx, axis=0)
        
        return data_item, label_item, name_item
    
    def pop_first_n(self, n):
        data_items = self.data[:n]
        label_items = self.labels[:n]
        name_items = self.name[:n]
        
        self.data = self.data[n:]
        self.labels = self.labels[n:]
        self.name = self.name[n:]
        
        return data_items, label_items, name_items
    def save(self, data_path, labels_path, names_path):
        np.save(data_path, self.data)
        np.save(labels_path, self.labels)
        with open(names_path, 'wb') as f:
            pickle.dump(self.name, f)
    def get_output_shape(self):
        return self.labels.shape[-1]

    @classmethod
    def load(cls, data_path, labels_path, names_path):
        data = np.load(data_path)
        labels = np.load(labels_path)
        with open(names_path, 'rb') as f:
            name = pickle.load(f)
        return cls(data, labels, name)