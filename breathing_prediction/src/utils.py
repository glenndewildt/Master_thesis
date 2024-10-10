import os
import math
import numpy as np
import pandas as pd
import scipy
import librosa
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from scipy.io import wavfile
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.model_selection import KFold
from dataset import *

class EarlyStopping:
    def __init__(self, patience=30, mode='min'):
        self.patience = patience
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0

    def step(self, current_score):
        if (self.mode == 'min' and current_score < self.best_score) or (self.mode == 'max' and current_score > self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            print(f"Early stopping counter {self.counter} epochs without improvement to de validation set.")


        if self.counter >= self.patience:
            print(f"Early stopping triggered after {self.counter} epochs without improvement.")
            return True
        return False

    
def prepare_data_model(audio_interspeech_norm, breath_interspeech_folder, window_size, step_size):
    # Load and prepare data
    train_data, train_labels, train_dict, frame_rate = load_data(audio_interspeech_norm, breath_interspeech_folder, 'train')
    devel_data, devel_labels, devel_dict, _ = load_data(audio_interspeech_norm, breath_interspeech_folder, 'devel')
    test_data, test_labels, test_dict, _ = load_data(audio_interspeech_norm, breath_interspeech_folder, 'test')
    
    # Prepare data
    prepared_train_data, prepared_train_labels, _ = prepare_data(train_data, train_labels, train_dict, frame_rate, window_size * 16000, step_size * 16000)
    prepared_devel_data, prepared_devel_labels, _ = prepare_data(devel_data, devel_labels, devel_dict, frame_rate, window_size * 16000, step_size * 16000)
    prepared_test_data, prepared_test_labels, _= prepare_data(test_data, test_labels, test_dict, frame_rate, window_size * 16000, step_size * 16000)

    # Create custom datasets
    test_dataset = CustomDataset(prepared_test_data, prepared_test_labels, test_dict.values())

    combined_train_data = np.concatenate((prepared_train_data, prepared_devel_data), axis=0)
    combined_train_labels = np.concatenate((prepared_train_labels, prepared_devel_labels), axis=0)
    combined_train_dict = list(train_dict.values()) + list(devel_dict.values())  
    combined_train_dataset = CustomDataset(combined_train_data, combined_train_labels, combined_train_dict)
    all_labels = pd.concat([test_labels, pd.concat([devel_labels, train_labels], axis=0)], axis=0)
    # Remove unused variables from memory
    del train_data, devel_data, test_data
    del prepared_train_data, prepared_devel_data, prepared_test_data, combined_train_data, combined_train_labels, combined_train_dict
    
    return combined_train_dataset, test_dataset, all_labels
    
def prepare_data_model_n_vall(audio_interspeech_norm, breath_interspeech_folder, window_size, step_size, num_val):
    # Load and prepare data
    train_data, train_labels, train_dict, frame_rate = load_data(audio_interspeech_norm, breath_interspeech_folder, 'train')
    devel_data, devel_labels, devel_dict, _ = load_data(audio_interspeech_norm, breath_interspeech_folder, 'devel')
    test_data, test_labels, test_dict, _ = load_data(audio_interspeech_norm, breath_interspeech_folder, 'test')
    
    # Prepare data
    prepared_train_data, prepared_train_labels, _ = prepare_data(train_data, train_labels, train_dict, frame_rate, window_size * 16000, step_size * 16000)
    prepared_devel_data, prepared_devel_labels, _ = prepare_data(devel_data, devel_labels, devel_dict, frame_rate, window_size * 16000, step_size * 16000)
    prepared_test_data, prepared_test_labels, _= prepare_data(test_data, test_labels, test_dict, frame_rate, window_size * 16000, step_size * 16000)

    # Create custom datasets
    train_dataset = CustomDataset(prepared_train_data, prepared_train_labels, train_dict)
    val_dataset = CustomDataset(prepared_devel_data, prepared_devel_labels, devel_dict)
    test_dataset = CustomDataset(prepared_test_data, prepared_test_labels, test_dict)
    train_dataset.print_shapes()
    val_dataset.print_shapes()
    print(num_val)
    new_val_dataset_item = val_dataset.pop_first_n(num_val)
    new_val_dataset = CustomDataset(new_val_dataset_item[0], new_val_dataset_item[1], new_val_dataset_item[2])
    new_val_dataset.print_shapes()

    combined_train_data = np.concatenate((train_dataset.data, val_dataset.data), axis=0)
    combined_train_labels = np.concatenate((train_dataset.labels, val_dataset.labels), axis=0)
    combined_train_dict = np.concatenate((train_dataset.name, val_dataset.name), axis=0)
    combined_train_data, combined_train_labels = flatten_data_for_model(combined_train_data, combined_train_labels)
    combined_train_dataset = CustomDataset(combined_train_data, combined_train_labels, [])
    all_labels = pd.concat([test_labels, pd.concat([devel_labels, train_labels], axis=0)], axis=0)

    return combined_train_dataset, new_val_dataset, test_dataset, all_labels

def load_data(path_to_data, path_to_labels, prefix):
    # labels
    labels=pd.read_csv(path_to_labels+'labels.csv', sep=',')
    labels = labels.loc[labels['filename'].str.contains(prefix)]
    labels['upper_belt']=labels['upper_belt'].astype('float32')

    # data
    fs, example = wavfile.read(path_to_data + labels.iloc[0, 0])
    result_data = np.zeros(shape=(np.unique(labels['filename']).shape[0], example.shape[0]))

    files=np.unique(labels['filename'])
    filename_dict={}
    for i in range(len(files)):
        frame_rate, data = wavfile.read(path_to_data+files[i])
        result_data[i]=data
        filename_dict[i]=files[i]
    return result_data, labels, filename_dict, frame_rate


def load_ucl_data(path_to_data, path_to_labels, seperator,sr, mfcc):
    # labels
    labels=pd.read_csv(path_to_labels+"labels.csv", sep=seperator)
    labels['filename'] = labels['filename'].str.replace("channel3", "channel1")
    labels = labels.loc[labels['filename'].str.contains("")]
    labels = labels.loc[labels['filename'].str.contains("test") == False]


    example , fs = librosa.load(path_to_data + labels.iloc[0, 0], sr=sr)

    result_data = np.zeros(shape=(np.unique(labels['filename']).shape[0], example.shape[0]))
    files = np.unique(labels['filename'])
    filename_dict = {}
    mfcc_results =[]
    for i in range(len(files)):
        
        if mfcc == True:
            # Load the WAV file with a resolution of 1000 Hz
            y, sr = librosa.load(path_to_data+files[i], sr=16000)
            
            # Calculate the number of samples per frame (n_fft)
            # This is based on the desired frequency resolution (25 Hz)
            n_fft = int(sr / 25) # n_fft = 40 samples per frame

            # Compute the MFCC, with 1 coefficient, a hop length of 10 ms, and a window length of 25 ms
            S = librosa.feature.mfcc(y=y, sr=sr, hop_length=int(0.010 * sr), n_fft=n_fft)

            # Convert the MFCC to log scale (dB)
            log_S = librosa.power_to_db(S, ref=np.max)

            # Append the result to the mfcc_results list
            mfcc_results.append(log_S)

            # Display the spectrogram
  # Plot the spectrogram
        
        frame_rate, data = wavfile.read(path_to_data+files[i])
        result_data[i]=data
        filename_dict[i]=files[i]
    return result_data, labels, filename_dict, frame_rate, mfcc_results




def how_many_windows_do_i_need(length_sequence, window_size, step):
    start_idx=0
    counter=0
    while True:
        if start_idx+window_size>length_sequence:
            break
        start_idx+=step
        counter+=1
    if start_idx!=length_sequence:
        counter+=1
    return counter


def flatten_data_for_model(data, labels):
    data_flattened = data.reshape(-1, data.shape[-1])
    labels_flattened = labels.reshape(-1, labels.shape[-1])
    return data_flattened, labels_flattened



def reshaping_data_for_model_test(data, labels):
    result_data = data.reshape( data.shape) 
    result_labels = labels.reshape(labels.shape) 
    return result_data, result_labels



def combine_last_two_dimensions(tensor):
    tensor = tensor.reshape(-1, tensor.shape[-1])
    return tensor

def flatten_and_shuffle_data(dataloader):
    all_inputs = []
    all_labels = []
    
    for inputs, labels, _ in dataloader:
        combined_inputs = combine_last_two_dimensions(inputs)
        combined_labels = combine_last_two_dimensions(labels)
        
        all_inputs.append(combined_inputs)
        all_labels.append(combined_labels)
    
    all_inputs = torch.cat(all_inputs)
    all_labels = torch.cat(all_labels)
    
    # Shuffle the data
    indices = torch.randperm(all_inputs.size(0))
    all_inputs = all_inputs[indices]
    all_labels = all_labels[indices]
    
    return all_inputs, all_labels

def prepare_data(data, labels, class_to_filename_dict, frame_rate, size_window, step_for_window):
    label_rate = 25  # 25 Hz label rate
    num_windows = how_many_windows_do_i_need(data.shape[1], size_window, step_for_window)
    new_data = np.zeros(shape=(data.shape[0], int(num_windows), size_window))
    length_of_label_window = int(size_window / frame_rate * label_rate)
    step_of_label_window = int(length_of_label_window * (step_for_window / size_window))
    new_labels = np.zeros(shape=(np.unique(labels['filename']).shape[0], int(num_windows), length_of_label_window))
    new_labels_timesteps = np.zeros(shape=new_labels.shape)

    for instance_idx in range(data.shape[0]):
        start_idx_data = 0
        start_idx_label = 0
        temp_labels = labels[labels['filename'] == class_to_filename_dict[instance_idx]]
        temp_labels = temp_labels.drop(columns=['filename'])
        temp_labels = temp_labels.values
        for windows_idx in range(num_windows - 1):
            new_data[instance_idx, windows_idx] = data[instance_idx, start_idx_data:start_idx_data + size_window]
            new_labels[instance_idx, windows_idx] = temp_labels[start_idx_label:start_idx_label + length_of_label_window, 1]
            new_labels_timesteps[instance_idx, windows_idx] = temp_labels[start_idx_label:start_idx_label + length_of_label_window, 0]
            start_idx_data += step_for_window
            start_idx_label += step_of_label_window
        if start_idx_data + size_window >= data.shape[1]:
            new_data[instance_idx, num_windows - 1] = data[instance_idx, data.shape[1] - size_window:data.shape[1]]
            new_labels[instance_idx, num_windows - 1] = temp_labels[temp_labels.shape[0] - length_of_label_window:temp_labels.shape[0], 1]
            new_labels_timesteps[instance_idx, num_windows - 1] = temp_labels[temp_labels.shape[0] - length_of_label_window:temp_labels.shape[0], 0]
        else:
            new_data[instance_idx, num_windows - 1] = data[instance_idx, start_idx_data:start_idx_data + size_window]
            new_labels[instance_idx, num_windows - 1] = temp_labels[start_idx_label:start_idx_label + length_of_label_window, 1]
            new_labels_timesteps[instance_idx, num_windows - 1] = temp_labels[start_idx_label:start_idx_label + length_of_label_window, 0]
            start_idx_data += step_for_window
            start_idx_label += step_of_label_window

    return new_data, new_labels, new_labels_timesteps




# Helper function to calculate the number of windows needed
def how_many_windows_do_i_need(data_length, size_window, step_for_window):
    return int(np.ceil((data_length - size_window) / step_for_window)) + 1


def correlation_coefficient_accuracy(y_true, y_pred):
    #squeezed_tensor = tf.squeeze(y_true, axis=-1)

    x = y_true
    y = y_pred
    mx = torch.mean(x, axis=1, keepdims=True)
    my = torch.mean(y, axis=1, keepdims=True)
    xm, ym = x - mx, y - my
    r_num = torch.sum(torch.multiply(xm, ym), axis=1)
    sum_square_x = torch.sum(torch.square(xm), axis=1)
    sum_square_y = torch.sum(torch.square(ym), axis=1)
    sqrt_x = torch.sqrt(sum_square_x)
    sqrt_y = torch.sqrt(sum_square_y)
    r_den = torch.multiply(sqrt_x, sqrt_y)
    r = torch.divide(r_num, r_den)
    # To avoid NaN in division, we handle the case when r_den is 0
    r = torch.where(torch.isnan(r), torch.zeros_like(r), r)
    return torch.mean(r)

def correlation_coefficient_loss(y_true, y_pred):
    s = y_true.shape
    t = y_pred.shape

    x=y_true
    y=y_pred
    mx=torch.mean(x, axis=1, keepdims=True)
    my=torch.mean(y, axis=1, keepdims=True)
    xm,ym=x-mx,y-my
    r_num=torch.sum(torch.multiply(xm, ym), axis=1)
    sum_square_x=torch.sum(torch.square(xm), axis=1)
    sum_square_y = torch.sum(torch.square(ym), axis=1)
    sqrt_x = torch.sqrt(sum_square_x)
    sqrt_y = torch.sqrt(sum_square_y)
    r_den=torch.multiply(sqrt_x, sqrt_y)
    result=torch.divide(r_num, r_den)
    #tf.print('result:', result)
    result=torch.mean(result)
    #tf.print('mean result:', result)
    return 1 - result

def concatenate_prediction(true_values, predicted_values, timesteps_labels, class_dict):
    predicted_values=predicted_values.reshape(timesteps_labels.shape)
    tmp=np.zeros(shape=(true_values.shape[0],3))
    result_predicted_values=pd.DataFrame(data=tmp, columns=true_values.columns, dtype='float32')
    result_predicted_values['filename']=result_predicted_values['filename'].astype('str')
    index_temp=0
    for instance_idx in range(predicted_values.shape[0]):
        timesteps=np.unique(timesteps_labels[instance_idx])
        for timestep in timesteps:
            # assignment for filename and timestep
            result_predicted_values.iloc[index_temp,0]=class_dict[instance_idx]
            result_predicted_values.iloc[index_temp,1]=timestep
            # calculate mean of windows
            result_predicted_values.iloc[index_temp,2]=np.mean(predicted_values[instance_idx,timesteps_labels[instance_idx]==timestep])
            index_temp+=1
        #print('concatenation...instance:', instance_idx, '  done')

    return result_predicted_values


