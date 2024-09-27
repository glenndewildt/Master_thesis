from torch.utils.data import Dataset
import pickle
import torch.nn as nn
import torch
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
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
            signal = self.apply_augmentation(signal)
        
        return signal, label

    def apply_augmentation(self, signal):
        # Ensure the signal is 2D (channels, data)
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        # List of all possible augmentations
        augmentations = [
            self.apply_noise,
            self.apply_gain,
            self.apply_pitch_shift,
            #self.apply_reverb,
            self.apply_lowpass,
            self.apply_highpass,
            self.apply_freq_mask,
            self.apply_time_mask,
        ]
        
        # Randomly apply 2-3 augmentations
        num_augmentations = random.randint(0, 1)
        chosen_augmentations = random.sample(augmentations, num_augmentations)
        
        for aug in chosen_augmentations:
            #print(f"{aug} shape {signal.shape}")
            signal = aug(signal)
        
        return signal.squeeze(0)  # Return to original shape

    def apply_noise(self, signal):
        # Generate noise and apply it with a specified SNR
        noise = torch.randn_like(signal)
        snr = torch.tensor([30])  # Example SNR value
        return F.add_noise(signal, noise, snr)

    def apply_gain(self, signal):
        gain_db = torch.tensor(random.uniform(-3, 3))  # Range of gain values in dB
        gain_db = torch.tensor(2)  # Example gain value in dB
        return F.gain(signal, gain_db)

    def apply_pitch_shift(self, signal):
        n_steps = random.uniform(-2, 2)
        return F.pitch_shift(signal, self.sample_rate, n_steps)

    def apply_reverb(self, signal):
        # Example impulse response (IR) for reverb
        ir = torch.randn(1, 4000)  # This should be replaced with a real IR
        return F.convolve(signal, ir)

    def apply_lowpass(self, signal):
        cutoff_freq = 4000  # Example cutoff frequency
        return F.lowpass_biquad(signal, self.sample_rate, cutoff_freq)

    def apply_highpass(self, signal):
        cutoff_freq = 100  # Example cutoff frequency
        return F.highpass_biquad(signal, self.sample_rate, cutoff_freq)

    def apply_freq_mask(self, signal):
        freq_mask_param = torch.tensor(400)  # Example frequency mask parameter
        return F.mask_along_axis(signal, freq_mask_param, 0, axis=0)

    def apply_time_mask(self, signal):
        time_mask_param = 20  # Example time mask parameter
        mask_value = 0  # Example mask value
        return F.mask_along_axis(signal, time_mask_param, mask_value, axis=1)
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