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
from transformers import Wav2Vec2Processor

class AugmentedDataset(Dataset):
    def __init__(self, data, labels, processor=None, sample_rate=16000, augment=True, device='cuda', wavml =True):
        self.data = torch.tensor(data) if not isinstance(data, torch.Tensor) else data
        self.labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        self.processor = processor if processor else None
        self.sample_rate = sample_rate
        self.augment = augment
        self.device = device
        self.wavml = wavml

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]

        if self.augment:
            signal = self.apply_data_augmentation(signal)

        return signal, label

    def collate_fn(self, batch):
        batch = batch
        signals, labels = zip(*batch)
        signals = torch.stack(signals).to('cuda')
        labels = torch.stack(labels).to('cuda')
        signals = signals.squeeze(dim=1)
        labels = labels.squeeze(dim=1)

        if self.processor is not None:
            # Process each item in the batch separately
            output = self.processor(
                signals.cpu().numpy(), 
                sampling_rate=self.sample_rate, 
                return_tensors="pt", 
                padding="longest"
            )

            # Stack the processed signals and attention masks
            input_values = output.input_values.to('cuda')
            
            # Check if 'attention_mask' is in the output dictionary
            if 'attention_mask' in output:
                attention_mask = output.attention_mask.to('cuda')
            else:
                # Create an attention mask filled with ones if not present
                attention_mask = torch.ones_like(input_values, dtype=torch.float32).to('cuda')

            del signals, output
        else:
            input_values = signals
            attention_mask = torch.ones_like(signals).to('cuda')
        
        return {"input_values": input_values, "attention_mask": attention_mask}, labels

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
        cutoff_freq = random.randint(2000, 8000)
        return F.lowpass_biquad(signal, self.sample_rate, cutoff_freq)

    def apply_highpass(self, signal):
        cutoff_freq = random.randint(20, 100)
        return F.highpass_biquad(signal, self.sample_rate, cutoff_freq)

    def apply_freq_mask(self, signal):
        freq_mask_param = random.randint(50, 300)
        return F.mask_along_axis(signal, freq_mask_param, 0, axis=1)

    def apply_time_mask(self, signal):
        time_mask_param = random.randint(10, 20)
        mask_value = 0
        return F.mask_along_axis(signal, time_mask_param, mask_value, axis=0)
    
class CustomDataset:
    def __init__(self, data, labels, name, processor=None, sample_rate=16000, augment=True, device='cuda', wavml=True):
        self.data = torch.tensor(data) if not isinstance(data, torch.Tensor) else data
        self.labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        print(name)
        self.name = np.array(list(name)) if not isinstance(name, np.ndarray) else name     
        self.processor = processor if processor else None
        self.sample_rate = sample_rate
        self.augment = augment
        self.device = device
        self.wavml = wavml
        self.seq_size = self.data.shape


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.name is not None and len(self.name) > 0:
            return self.data[idx], self.labels[idx], self.name[idx]
        else:
            return self.data[idx], self.labels[idx], "none"

    def collate_fn(self, batch):
        signals, labels, name = zip(*batch)
        signals = torch.stack(signals)
        labels = torch.stack(labels).to(self.device)
        name = np.stack(name)
        
        if self.processor is not None:
            # Process each item in the batch separately
            
            processed_signals = []
            attention_masks = []
            for i in range(signals.shape[0]):
                output = self.processor(
                    signals[i].cpu().numpy(), 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt", 
                    padding="longest"
                )
                
                processed_signals.append(output.input_values)
                if 'attention_mask' in output:
                    attention_mask = output.attention_mask
                else:
                # Create an attention mask filled with ones if not present
                    attention_mask = torch.ones_like(output.input_values)
                    
                attention_masks.append(attention_mask)

            # Stack the processed signals and attention masks
            input_values = torch.stack(processed_signals)
            attention_mask = torch.stack(attention_masks)
            

            
            del signals, output
        else:
            input_values = signals
            attention_mask = torch.ones_like(signals)
        
        return {"input_values": input_values, "attention_mask": attention_mask}, labels, name
        
    def input_values(self):
        return self.seq_size

    def print_shapes(self):
        print(f"data: {self.data.shape}, labels: {self.labels.shape}, names: {self.name.shape}")

    def pop(self, idx):
        data_item = self.data[idx]
        label_item = self.labels[idx]
        name_item = self.name[idx]

        self.data = torch.cat((self.data[:idx], self.data[idx+1:]), dim=0)
        self.labels = torch.cat((self.labels[:idx], self.labels[idx+1:]), dim=0)
        self.name = torch.cat((self.name[:idx], self.name[idx+1:]), dim=0)

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