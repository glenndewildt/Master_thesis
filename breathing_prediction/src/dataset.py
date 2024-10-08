from torch.utils.data import Dataset
import pickle
import torch.nn as nn
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset
import torchaudio.functional as F
import random

class AugmentedDataset(Dataset):
    def __init__(self, data, labels, sample_rate=16000, augment=True):
        self.data = torch.tensor(data) if not isinstance(data, torch.Tensor) else data
        self.labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        self.sample_rate = sample_rate
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]

        if self.augment:
            signal = self.apply_data_augmentation(signal)

        return signal, label
    
    def apply_data_augmentation(self, signal, noise_prob=0.1, gain_prob=0.1, pitch_shift_prob=0.1, 
                                lowpass_prob=0.1, highpass_prob=0.1, freq_mask_prob=0.1, time_mask_prob=0.1):
        augmented_signal = signal.clone()
        if augmented_signal.dim() == 1:
            augmented_signal = augmented_signal.unsqueeze(0)
        
        if random.random() < noise_prob:
            augmented_signal = self.apply_noise(augmented_signal)
        
        if random.random() < gain_prob:
            augmented_signal = self.apply_gain(augmented_signal)
        
        if random.random() < pitch_shift_prob:
            augmented_signal = self.apply_pitch_shift(augmented_signal)
        
        if random.random() < lowpass_prob:
            augmented_signal = self.apply_lowpass(augmented_signal)
        
        if random.random() < highpass_prob:
            augmented_signal = self.apply_highpass(augmented_signal)
        
        if random.random() < freq_mask_prob:
            augmented_signal = self.apply_freq_mask(augmented_signal)
        
        if random.random() < time_mask_prob:
            augmented_signal = self.apply_time_mask(augmented_signal)
        
        return augmented_signal.squeeze(0)



    def apply_noise(self, signal):
        noise = torch.randn_like(signal)
        snr = torch.tensor([random.randint(20, 30)])  # Subtle noise (high SNR)
        return F.add_noise(signal, noise, snr)

    def apply_gain(self, signal):
        gain_db = torch.tensor(random.randint(-2, 2))  # Subtle gain adjustment
        return F.gain(signal, gain_db)

    def apply_pitch_shift(self, signal):
        n_steps = random.randint(-2, 2)  # Subtle pitch shift
        return F.pitch_shift(signal, self.sample_rate, n_steps)

    def apply_lowpass(self, signal):
        cutoff_freq = random.randint(2000, 8000)  # Subtle lowpass (high cutoff)
        return F.lowpass_biquad(signal, self.sample_rate, cutoff_freq)

    def apply_highpass(self, signal):
        cutoff_freq = random.randint(20, 100)  # Subtle highpass (low cutoff)
        return F.highpass_biquad(signal, self.sample_rate, cutoff_freq)

    def apply_freq_mask(self, signal):
        freq_mask_param = torch.tensor(random.randint(50, 300))  # Narrow frequency mask
        return F.mask_along_axis(signal, freq_mask_param, 0, axis=1)

    def apply_time_mask(self, signal):
        time_mask_param = random.randint(10, 20)  # Short time mask
        mask_value = 0
        return F.mask_along_axis(signal, time_mask_param, mask_value, axis=0)
    
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