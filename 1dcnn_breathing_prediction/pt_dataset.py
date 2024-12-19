import torch
from torch.utils.data import Dataset
import numpy as np
import torchaudio
from typing import List, Tuple
import random
from transformers import Wav2Vec2Processor
import torchaudio.functional as F
from torch.cuda.amp import autocast, GradScaler

class BreathingDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 processor: Wav2Vec2Processor, window_size: int, step_size: int, 
                 sample_rate: int = 16000, augment: bool = False):
        self.data = data
        self.labels = labels
        self.processor = processor
        self.window_size = window_size
        self.step_size = step_size
        self.sample_rate = sample_rate
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio = torch.from_numpy(self.data[idx]).float()
        label = torch.from_numpy(self.labels[idx]).float()

        if self.augment:
            audio = self.apply_augmentation(audio)

        return audio, label
    
    def apply_data_augmentation(self, signal):
        augmented_signal = signal.clone()
        if augmented_signal.dim() == 1:
            augmented_signal = augmented_signal.unsqueeze(0)

        aug_functions = [
            (0.1, self.apply_noise),
            (0.1, self.apply_gain),
            (0.1, self.apply_pitch_shift),
            (0.1, self.apply_lowpass),
            (0.1, self.apply_highpass),
            (0.1, self.apply_freq_mask),
            (0.1, self.apply_time_mask)
        ]

        for prob, func in aug_functions:
            if random.random() < prob:
                augmented_signal = func(augmented_signal)

        return augmented_signal.squeeze(0)

    def apply_noise(self, signal):
        noise = torch.randn_like(signal)
        snr = torch.tensor([random.randint(20, 30)])
        return F.add_noise(signal, noise, snr)

    def apply_gain(self, signal):
        gain_db = torch.tensor(random.randint(-2, 2))
        return F.gain(signal, gain_db)

    def apply_pitch_shift(self, signal):
        n_steps = random.randint(-2, 2)
        return F.pitch_shift(signal, self.sample_rate, n_steps)

    def apply_lowpass(self, signal):
        cutoff_freq = random.randint(4000, 8000)
        return F.lowpass_biquad(signal, self.sample_rate, cutoff_freq)

    def apply_highpass(self, signal):
        cutoff_freq = random.randint(50, 100)
        return F.highpass_biquad(signal, self.sample_rate, cutoff_freq)

    def apply_freq_mask(self, signal):
        freq_mask_param = random.randint(20, 50)
        return F.mask_along_axis(signal, freq_mask_param, 0, axis=1)

    def apply_time_mask(self, signal):
        time_mask_param = random.randint(5, 10)
        mask_value = 0
        return F.mask_along_axis(signal, time_mask_param, mask_value, axis=0)

    def apply_augmentation(self, audio: torch.Tensor):
        audio = self.apply_data_augmentation(audio)
        
        return audio
    
    def collate_fn(self, batch):
        audios, labels = zip(*batch)
        audios = torch.stack(audios)
        labels = torch.stack(labels)
        
        
        # Process with Wav2Vec2 processor
        inputs = self.processor(
            audios.numpy(), 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        return inputs, labels

    @classmethod
    def create_dataloader(cls, data: np.ndarray, labels: np.ndarray, 
                          processor: Wav2Vec2Processor, window_size: int, step_size: int, 
                          batch_size: int, sample_rate: int = 16000, augment: bool = False, 
                          shuffle: bool = False, num_workers: int = 0):
        dataset = cls(data, labels, processor, window_size, step_size, sample_rate, augment)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda batch: cls.collate_fn(batch, processor),
            pin_memory=True
        )


class GPUBreathingDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 processor: Wav2Vec2Processor, 
                 sample_rate: int = 16000, augment: bool = False):
        self.data = torch.from_numpy(data).float().cuda()
        self.labels = torch.from_numpy(labels).float().cuda()
        self.processor = processor
        self.sample_rate = sample_rate
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio = self.data[idx]
        label = self.labels[idx]

        if self.augment:
            audio = self.apply_augmentation(audio)

        return audio, label
    
    def apply_data_augmentation(self, signal):
        augmented_signal = signal.clone()
        if augmented_signal.dim() == 1:
            augmented_signal = augmented_signal.unsqueeze(0)

        aug_functions = [
            (0.2, self.apply_noise),
            (0.2, self.apply_gain),
            (0.2, self.apply_pitch_shift),
            #(0.1, self.apply_lowpass),
            #(0.1, self.apply_highpass),
            (0.1, self.apply_freq_mask),
            (0.1, self.apply_time_mask)
        ]

        for prob, func in aug_functions:
            if random.random() < prob:
                augmented_signal = func(augmented_signal)

        return augmented_signal.squeeze(0)

    def apply_noise(self, signal):
        noise = torch.randn_like(signal).cuda()
        snr = torch.tensor([random.randint(20, 30)], device='cuda')
        return F.add_noise(signal, noise, snr)

    def apply_gain(self, signal):
        gain_db = torch.tensor(random.randint(-2, 2), device='cuda')
        return F.gain(signal, gain_db)

    def apply_pitch_shift(self, signal):
        n_steps = random.randint(-2, 2)
        return F.pitch_shift(signal, self.sample_rate, n_steps)

    def apply_lowpass(self, signal):
        cutoff_freq = random.randint(4000, 8000)
        return F.lowpass_biquad(signal, self.sample_rate, cutoff_freq)

    def apply_highpass(self, signal):
        cutoff_freq = random.randint(50, 100)
        return F.highpass_biquad(signal, self.sample_rate, cutoff_freq)

    def apply_freq_mask(self, signal):
        freq_mask_param = random.randint(20, 50)
        return F.mask_along_axis(signal, freq_mask_param, 0, axis=1)

    def apply_time_mask(self, signal):
        time_mask_param = random.randint(5, 10)
        mask_value = 0
        return F.mask_along_axis(signal, time_mask_param, mask_value, axis=0)

    def apply_augmentation(self, audio: torch.Tensor):
        audio = self.apply_data_augmentation(audio)
        return audio
    
    def collate_fn(self, batch):
        audios, labels = zip(*batch)
        audios = torch.stack(audios)
        labels = torch.stack(labels)
        
        # Process with Wav2Vec2 processor
        inputs = self.processor(
            audios.cpu().numpy(),  # Processor expects numpy array
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move inputs to GPU
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        return inputs, labels

    @classmethod
    def create_dataloader(cls, data: np.ndarray, labels: np.ndarray, 
                          processor: Wav2Vec2Processor, window_size: int, step_size: int, 
                          batch_size: int, sample_rate: int = 16000, augment: bool = False, 
                          shuffle: bool = False, num_workers: int = 0):
        dataset = cls(data, labels, processor, window_size, step_size, sample_rate, augment)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=False  # Already on GPU, so no need for pin_memory
        )