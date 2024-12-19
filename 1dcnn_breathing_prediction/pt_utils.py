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
from pt_dataset import *
from scipy import interpolate
from scipy.signal import gauss_spline
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, sosfilt



class EarlyStopping:
    def __init__(self, patience=7, mode='min', delta=0):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            if score < self.best_score - self.delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

def unsplit_data_ogsize(windowed_data, window_size, step_size, data_points_per_second, original_length):
    # Convert to a PyTorch tensor and move to GPU

    # if isinstance(windowed_data, np.ndarray):
    #     windowed_data = tf.Tensor(windowed_data)  # Use 'cuda' for GPU

    window_size_points = window_size * data_points_per_second
    step_size_points = step_size * data_points_per_second
    batch_size, num_windows, data_lenght = windowed_data.shape
    
    original_data = np.zeros((batch_size, original_length))
    overlap_count = np.zeros((batch_size, original_length))

    def process_batch(batch_index):
        for i in range(num_windows):
            start = i * step_size_points
            end = start + window_size_points

            segment_length = end - start

            # Update original data and overlap count
            original_data[batch_index, start:end] += windowed_data[batch_index, i, :segment_length]
            overlap_count[batch_index, start:end] += 1

    # Use Torch's built-in parallelism
    for b in range(batch_size):
        process_batch(b)

    # Average the overlapping regions
    original_data = np.where(overlap_count != 0, original_data / overlap_count, np.zeros_like(original_data))

    # Trim the data to match the original length
    original_data = original_data[:, :original_length]

    return original_data

def average_overlap(predictions, window_size, step_size):
    total_length = (len(predictions) - 1) * step_size + window_size
    reconstructed = np.zeros(total_length)
    counts = np.zeros(total_length)
    
    for i, pred in enumerate(predictions):
        start = i * step_size
        end = start + window_size
        reconstructed[start:end] += pred
        counts[start:end] += 1
    
    return reconstructed / counts

def linear_interpolation(predictions, window_size, step_size):
    centers = np.arange(window_size // 2, len(predictions) * step_size, step_size)
    total_length = (len(predictions) - 1) * step_size + window_size
    x_new = np.arange(total_length)
    
    center_values = predictions[:, window_size // 2]
    f = interpolate.interp1d(centers, center_values, kind='linear', fill_value='extrapolate')
    
    return f(x_new)

def weighted_average(predictions, window_size, step_size):
    total_length = (len(predictions) - 1) * step_size + window_size
    reconstructed = np.zeros(total_length)
    weights = np.zeros(total_length)
    
    window_weights = 1 - np.abs(np.linspace(-1, 1, window_size))
    
    for i, pred in enumerate(predictions):
        start = i * step_size
        end = start + window_size
        reconstructed[start:end] += pred * window_weights
        weights[start:end] += window_weights
    
    return reconstructed / weights
def last_valid_prediction(predictions, window_size, step_size):
    total_length = (len(predictions) - 1) * step_size + window_size
    reconstructed = np.zeros(total_length)
    
    for i, pred in enumerate(reversed(predictions)):
        start = total_length - (i + 1) * step_size
        end = start + window_size
        reconstructed[start:end] = pred
    
    return reconstructed

def gaussian_weighted(predictions, window_size, step_size):
    total_length = (len(predictions) - 1) * step_size + window_size
    reconstructed = np.zeros(total_length)
    weights = np.zeros(total_length)
    
    gaussian_window = gauss_spline(window_size, std=window_size/6)
    
    for i, pred in enumerate(predictions):
        start = i * step_size
        end = start + window_size
        reconstructed[start:end] += pred * gaussian_window
        weights[start:end] += gaussian_window
    
    return reconstructed / weights

def concatenate_prediction_og(predicted_values, timesteps_labels, class_dict, columns_for_real_labels=['filename', 'timeFrame', 'upper_belt']):
    predicted_values = predicted_values.reshape(timesteps_labels.shape)
    result_predicted_values = pd.DataFrame(columns=columns_for_real_labels, dtype='float32')
    result_predicted_values['filename'] = result_predicted_values['filename'].astype('str')
    
    for instance_idx in range(predicted_values.shape[0]):
        predicted_values_tmp = pd.DataFrame(predicted_values[instance_idx], columns=['upper_belt'])
        timesteps_labels_tmp = pd.DataFrame(timesteps_labels[instance_idx], columns=['timeFrame'])
        
        # Round timeFrame to 4 decimal places for comparison
        timesteps_labels_tmp['timeFrame'] = timesteps_labels_tmp['timeFrame'].round(4)
        
        tmp = pd.merge(timesteps_labels_tmp, predicted_values_tmp, left_index=True, right_index=True)
        tmp = tmp.groupby(by=['timeFrame']).mean().reset_index()
        tmp['filename'] = class_dict[instance_idx]
        
        result_predicted_values = pd.concat([result_predicted_values, tmp.copy(deep=True)], ignore_index=True)
    
    result_predicted_values['timeFrame'] = result_predicted_values['timeFrame'].astype('float32')
    result_predicted_values['upper_belt'] = result_predicted_values['upper_belt'].astype('float32')
    
    return result_predicted_values['upper_belt']

def reshaping_data_for_model(data, labels):
    result_data=data.reshape((-1,data.shape[2]))
    result_labels=labels.reshape((-1,labels.shape[2]))
    return result_data, result_labels
    
def prepare_data_model_fold(audio_interspeech_norm, breath_interspeech_folder, window_size, step_size):
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
    train_dataset = AugmentedDataset(prepared_train_data.reshape(-1, prepared_train_data.shape[-1]), prepared_train_labels.reshape(-1, prepared_train_labels.shape[-1]), augment=True)
    develop_dataset = CustomDataset(prepared_devel_data, prepared_devel_labels, devel_dict.values())

    all_labels = pd.concat([test_labels, pd.concat([devel_labels, train_labels], axis=0)], axis=0)
    # Remove unused variables from memory
    del train_data, devel_data, test_data
    del prepared_train_data, prepared_devel_data, prepared_test_data
    
    return train_dataset, develop_dataset, test_dataset, all_labels
    

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
    r=torch.divide(r_num, r_den)
    #tf.print('result:', result)
    r=torch.mean(r)
    #tf.print('mean result:', result)
    r = torch.where(torch.isnan(r), torch.zeros_like(r), r)

    return 1 - r

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

def unsplit_data(windowed_data,window_size, step_size, method, original_length, data_points_per_second= 25):

    if method == 'original':
        return unsplit_data_ogsize(windowed_data, window_size, step_size, data_points_per_second, original_length)
    elif method == 'gaussian':
        return unsplit_data_gaussian(windowed_data, window_size, step_size, data_points_per_second, original_length)
    #elif method == 'cubic':
        #return unsplit_data_advanced(windowed_data, window_size, step_size, data_points_per_second, original_length)
    elif method == 'kalman':
        return unsplit_data_kalman(windowed_data, window_size, step_size, data_points_per_second, original_length)
    else:
        raise ValueError(f"Unknown method: {method}")

def unsplit_data_ogsize(windowed_data, window_size, step_size, data_points_per_second, original_length):
    # Convert to a PyTorch tensor and move to GPU

    if isinstance(windowed_data, np.ndarray):
        windowed_data = torch.tensor(windowed_data, device='cuda')  # Use 'cuda' for GPU

    device = windowed_data.device  # Ensure to use the same device as windowed_data     
    window_size_points = window_size * data_points_per_second
    step_size_points = step_size * data_points_per_second
    batch_size, num_windows, data_lenght = windowed_data.shape
    
    original_data = torch.zeros((batch_size, original_length), device=device)
    overlap_count = torch.zeros((batch_size, original_length), device=device)

    def process_batch(batch_index):
        for i in range(num_windows):
            start = i * step_size_points
            end = start + window_size_points
            if end > original_length:
                end = original_length
            segment_length = end - start

            # Update original data and overlap count
            original_data[batch_index, start:end] += windowed_data[batch_index, i, :segment_length]
            overlap_count[batch_index, start:end] += 1

    # Use Torch's built-in parallelism
    for b in range(batch_size):
        process_batch(b)

    # Average the overlapping regions
    with torch.no_grad():
        original_data = torch.where(overlap_count > 0, original_data / overlap_count, torch.zeros_like(original_data))

    # Trim the data to match the original length
    original_data = original_data[:, :original_length]

    return original_data.to("cpu")

def unsplit_data_last(windowed_data, window_size, step_size, data_points_per_second, original_length):
    # Convert to a PyTorch tensor and move to GPU

    if isinstance(windowed_data, np.ndarray):
        windowed_data = torch.tensor(windowed_data, device='cuda')  # Use 'cuda' for GPU

    device = windowed_data.device  # Ensure to use the same device as windowed_data     
    window_size_points = window_size * data_points_per_second
    step_size_points = step_size * data_points_per_second
    batch_size, num_windows, data_lenght = windowed_data.shape
    
    original_data = torch.zeros((batch_size, original_length), device=device)
    overlap_count = torch.zeros((batch_size, original_length), device=device)

    def process_batch(batch_index):
        for i in range(num_windows):
            start = i * step_size_points
            end = start + window_size_points
            if end > original_length:
                end = original_length
            segment_length = end - start

            # Update original data and overlap count
             
            original_data[batch_index, start:end] = windowed_data[batch_index, i, :segment_length]
            overlap_count[batch_index, start:end] += 1

    # Use Torch's built-in parallelism
    for b in range(batch_size):
        process_batch(b)


    # Trim the data to match the original length
    original_data = original_data[:, :original_length]

    return original_data.to("cpu")
from collections import Counter

def unsplit_data_mean(windowed_data, window_size, step_size, data_points_per_second, original_length):
    # Convert to a PyTorch tensor and move to GPU
    if isinstance(windowed_data, np.ndarray):
        windowed_data = torch.tensor(windowed_data, device='cuda')

    device = windowed_data.device
    window_size_points = window_size * data_points_per_second
    step_size_points = step_size * data_points_per_second
    batch_size, num_windows, data_length = windowed_data.shape
    
    original_data = torch.zeros((batch_size, original_length), device=device)
    overlap_values = [[] for _ in range(original_length)]

    def process_batch(batch_index):
        for i in range(num_windows):
            start = i * step_size_points
            end = start + window_size_points
            if end > original_length:
                end = original_length
            segment_length = end - start

            # Update original data and keep track of values
            for j in range(start, end):
                value = windowed_data[batch_index, i, j - start]
                original_data[batch_index, j] += value
                overlap_values[j].append(value)

    # Use Torch's built-in parallelism
    for b in range(batch_size):
        process_batch(b)

    # Choose the most common value for each overlapping region
    for j in range(original_length):
        if overlap_values[j]:
            most_common = Counter(overlap_values[j]).most_common(1)[0][0]
            original_data[:, j] = most_common

    # Trim the data to match the original length
    original_data = original_data[:, :original_length]

    return original_data.to("cpu")

def unsplit_data_gaussian(windowed_data, window_size, step_size, data_points_per_second, original_length):

    batch_size, num_windows, prediction_size = windowed_data.shape
    window_size_points = int(window_size * data_points_per_second)
    step_size_points = int(step_size * data_points_per_second)
    original_data = np.zeros((batch_size, original_length))
    overlap_count = np.zeros((batch_size, original_length))
    
    for b in range(batch_size):
        for i in range(num_windows):
            start = i * step_size_points
            end = min(start + window_size_points, original_length)
            segment_length = end - start
            original_data[b, start:end] += windowed_data[b, i, :segment_length]
            overlap_count[b, start:end] += 1
    
    # Average the overlapping regions
    original_data = np.divide(original_data, overlap_count, where=overlap_count != 0)
    
    # Apply Gaussian filter to the entire signal
    for b in range(batch_size):
        original_data[b] = gaussian_filter(original_data[b], sigma=.5)
    
    return original_data

def unsplit_data_cubic(windowed_data, window_size, step_size, data_points_per_second, original_length):

    batch_size, num_windows, prediction_size = windowed_data.shape
    window_size_points = int(window_size * data_points_per_second)
    step_size_points = int(step_size * data_points_per_second)
    original_data = np.zeros((batch_size, original_length))
    overlap_count = np.zeros((batch_size, original_length))
    
    for b in range(batch_size):
        for i in range(num_windows):
            start = i * step_size_points
            end = min(start + window_size_points, original_length)
            segment_length = end - start
            original_data[b, start:end] += windowed_data[b, i, :segment_length]
            overlap_count[b, start:end] += 1
    
    # Average the overlapping regions
    original_data = np.divide(original_data, overlap_count, where=overlap_count != 0)
    
    # Apply cubic spline interpolation to the entire signal
    for b in range(batch_size):
        x = np.arange(original_length)
        cs = CubicSpline(x, original_data[b])
        original_data[b] = cs(x)
    
    return original_data

import torch

import torch
import torch.nn.functional as F

class BreathingKalmanFilter:
    def __init__(self, initial_state_mean, state_covariance, observation_covariance):
        self.initial_state_mean = torch.tensor(initial_state_mean, dtype=torch.float32)
        self.state_covariance = torch.tensor(state_covariance, dtype=torch.float32)
        self.observation_covariance = torch.tensor(observation_covariance, dtype=torch.float32)
        self.current_state_estimate = self.initial_state_mean
        self.prediction_error = torch.tensor(0.0, dtype=torch.float32)
        self.current_velocity = torch.tensor(0.0, dtype=torch.float32)
        self.velocity_covariance = torch.tensor(0.1, dtype=torch.float32)
        
        # Add exponential decay factor for velocity
        self.velocity_decay = torch.tensor(0.95, dtype=torch.float32)
        # Add confidence weight for transitions
        self.transition_confidence = torch.tensor(0.8, dtype=torch.float32)

    def predict(self):
        # Add momentum term to smooth transitions
        momentum = self.current_velocity * self.transition_confidence
        self.current_state_estimate = self.current_state_estimate + momentum
        
        # Constrain to [-1, 1] with smooth clamping
        self.current_state_estimate = torch.tanh(self.current_state_estimate)
        
        # Update covariances with adaptive terms
        velocity_magnitude = torch.abs(self.current_velocity)
        adaptive_factor = torch.exp(-velocity_magnitude)
        self.state_covariance = self.state_covariance + self.observation_covariance * adaptive_factor
        self.velocity_covariance = self.velocity_covariance * (1.0 + 0.1 * adaptive_factor)

    def update(self, observation):
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        
        # Calculate innovation with confidence weighting
        innovation = observation - self.current_state_estimate
        confidence = torch.exp(-torch.abs(innovation))
        
        # Adaptive Kalman gains based on confidence
        state_gain = self.state_covariance / (self.state_covariance + self.observation_covariance)
        state_gain = state_gain * confidence
        
        velocity_gain = self.velocity_covariance / (self.velocity_covariance + self.observation_covariance)
        velocity_gain = velocity_gain * confidence
        
        # Update state estimate with momentum
        self.current_state_estimate = self.current_state_estimate + state_gain * innovation
        self.current_state_estimate = torch.tanh(self.current_state_estimate)
        
        # Update velocity with decay
        self.current_velocity = (self.current_velocity + velocity_gain * innovation) * self.velocity_decay
        
        # Update prediction error with confidence weighting
        self.prediction_error = self.prediction_error + innovation.abs() * confidence
        
        # Update covariances with adaptive terms
        self.state_covariance = self.state_covariance * (1 - state_gain * confidence)
        self.velocity_covariance = self.velocity_covariance * (1 - velocity_gain * confidence)

    def filter(self, data):
        """Filters the data using predict and update iteratively"""
        estimates = []
        errors = []
        for observation in data:
            self.predict()
            self.update(observation)
            estimates.append(self.current_state_estimate)
            errors.append(self.prediction_error)
        return torch.stack(estimates), torch.stack(errors)

def process_overlap_region(data1, data2, start_idx, end_idx, params_list):
    """Process overlapping region using multiple Kalman filters with different parameters."""
    best_estimate = None
    min_error = float('inf')
    
    # Ensure data2 is the correct length
    segment_length = end_idx - start_idx
    if len(data2) > segment_length:
        data2 = data2[:segment_length]
    elif len(data2) < segment_length:
        padding = torch.full((segment_length - len(data2),), data2[-1], device=data2.device)
        data2 = torch.cat([data2, padding])
    
    # Calculate weights using exponential moving average of stability
    weights = torch.ones(segment_length, device=data1.device)
    alpha = 0.3  # smoothing factor
    
    for i in range(1, segment_length):
        stability1 = torch.exp(-torch.abs(data1[start_idx + i] - data1[start_idx + i - 1]))
        stability2 = torch.exp(-torch.abs(data2[i] - data2[i - 1]))
        
        # Smooth the weights
        weights[i] = alpha * (stability1 / (stability1 + stability2)) + (1 - alpha) * weights[i - 1]
    
    # Apply sigmoid to get smoother transitions
    weights = torch.sigmoid(3 * (weights - 0.5))
    
    overlap_data = data1[start_idx:end_idx] * weights + data2 * (1 - weights)
    
    for params in params_list:
        kf = BreathingKalmanFilter(
            initial_state_mean=params['initial_state'],
            state_covariance=params['state_cov'],
            observation_covariance=params['obs_cov']
        )
        
        # Move parameters to device
        for attr in ['current_state_estimate', 'state_covariance', 'observation_covariance', 
                    'current_velocity', 'velocity_covariance', 'velocity_decay', 'transition_confidence']:
            setattr(kf, attr, getattr(kf, attr).to(overlap_data.device))
        
        estimates, error = kf.filter(overlap_data)
        
        if error[-1] < min_error:  # Use the final error to select the best filter
            min_error = error[-1]
            best_estimate = estimates
            
    return best_estimate

def unsplit_data_kalman(windowed_data, window_size, step_size, data_points_per_second, original_length):
    windowed_data = torch.tensor(windowed_data, device='cuda')
    batch_size, num_windows, prediction_size = windowed_data.shape
    
    window_size_points = int(window_size * data_points_per_second)
    step_size_points = int(step_size * data_points_per_second)
    
    original_data = torch.zeros((batch_size, original_length), device='cuda')
    overlap_count = torch.zeros((batch_size, original_length), device='cuda')
    
    # Expanded parameter set for different breathing patterns
    kalman_params_list = [
        {'initial_state': 0.0, 'state_cov': 0.1, 'obs_cov': 0.05},  # Fast breathing
        {'initial_state': 0.0, 'state_cov': 0.3, 'obs_cov': 0.1},   # Normal breathing
        {'initial_state': 0.0, 'state_cov': 0.5, 'obs_cov': 0.2},   # Slow breathing
        {'initial_state': 0.0, 'state_cov': 0.2, 'obs_cov': 0.15}   # Transitional
    ]
    
    for b in range(batch_size):
        for i in range(num_windows):
            start = i * step_size_points
            end = min(start + window_size_points, original_length)
            current_window = windowed_data[b, i, :(end - start)]
            
            if overlap_count[b, start:end].max() == 0:
                original_data[b, start:end] = current_window
                overlap_count[b, start:end] = 1
            else:
                overlap_estimate = process_overlap_region(
                    original_data[b],
                    current_window,
                    start,
                    end,
                    kalman_params_list
                )
                # Smooth transition at boundaries
                if i > 0:
                    overlap_estimate[0:5] = torch.lerp(
                        original_data[b, start:start + 5],
                        overlap_estimate[0:5],
                        torch.linspace(0, 1, 5, device='cuda')
                    )
                original_data[b, start:end] = overlap_estimate

    # Final smoothing with adaptive parameters
    for b in range(batch_size):
        # Calculate signal statistics for adaptive smoothing
        signal_std = original_data[b].std()
        final_kf = BreathingKalmanFilter(
            initial_state_mean=0.0,
            state_covariance=0.2 * signal_std,
            observation_covariance=0.1 * signal_std
        )
        
        # Move filter to GPU
        for attr in ['current_state_estimate', 'state_covariance', 'observation_covariance', 
                    'current_velocity', 'velocity_covariance', 'velocity_decay', 'transition_confidence']:
            setattr(final_kf, attr, getattr(final_kf, attr).to(original_data.device))
        
        smoothed_estimates, _ = final_kf.filter(original_data[b])
        original_data[b] = torch.tanh(smoothed_estimates)  # Smooth clamping
    return original_data.cpu()

import torch
import torch.nn.functional as F

class WaveletBasedReconstruction:
    def __init__(self, wave_type='db4', level=3):
        self.wave_type = wave_type
        self.level = level
        
    def _get_wavelet_kernels(self):
        """Generate wavelet filter kernels based on the wavelet type (e.g., 'db4')"""
        # In practice, you would use PyWavelets or similar library to get wavelet filters
        # For simplicity, we provide the 'db4' filters (Daubechies 4-tap wavelet)
        if self.wave_type == 'db4':
            # Define the 'db4' low-pass and high-pass filter coefficients
            low_pass = np.array([0.48296, 0.8365, 0.22414, -0.12940])
            high_pass = np.array([-0.1294, -0.22414, 0.8365, -0.48296])
        else:
            raise ValueError(f"Wavelet type {self.wave_type} not supported. Please choose 'db4'.")
        
        # Convert to torch tensors
        low_pass = torch.tensor(low_pass, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
        high_pass = torch.tensor(high_pass, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
        
        return {'low': low_pass, 'high': high_pass}
    
    def decompose(self, signal):
        """Simulate wavelet decomposition using filterbanks"""
        kernels = self._get_wavelet_kernels()
        coeffs = []
        
        x = signal
        for i in range(self.level):
            # Apply convolutions for decomposition using low-pass and high-pass filters
            x_low = F.conv1d(x.unsqueeze(1), kernels['low'], stride=2)  # Downsample by 2
            x_high = F.conv1d(x.unsqueeze(1), kernels['high'], stride=2)  # Downsample by 2
            
            coeffs.append((x_low.squeeze(1), x_high.squeeze(1)))  # Remove extra channel dim
            x = x_low  # Continue decomposition with low-pass signal
        
        return coeffs
    
    def reconstruct_overlap(self, data1, data2, overlap_region):
        """Blend signals in wavelet domain"""
        coeffs1 = self.decompose(data1)
        coeffs2 = self.decompose(data2)
        
        # Blend coefficients based on correlation in each subband
        blended_coeffs = []
        for c1, c2 in zip(coeffs1, coeffs2):
            correlation = F.cosine_similarity(c1[0].unsqueeze(0), c2[0].unsqueeze(0))  # Similarity for low-pass
            weight = torch.sigmoid(correlation)  # Get blend weight from similarity
            blended = c1[0] * weight + c2[0] * (1 - weight)  # Blend based on weight
            blended_coeffs.append(blended)
        
        return self._reconstruct(blended_coeffs)
    
    def _reconstruct(self, coeffs):
        """Reconstruct the signal from blended wavelet coefficients"""
        # Start with the final set of coefficients
        x_reconstructed = coeffs[-1]  # Start from the last level
        
        # Iterate through the coefficients in reverse order
        for c in reversed(coeffs[:-1]):
            # Upsample (reconstruct) using inverse wavelet transform
            low_pass_reconstructed = F.conv_transpose1d(x_reconstructed.unsqueeze(1), self._get_wavelet_kernels()['low'], stride=2)
            high_pass_reconstructed = F.conv_transpose1d(x_reconstructed.unsqueeze(1), self._get_wavelet_kernels()['high'], stride=2)
            
            # Combine the low and high pass components (approximate inverse wavelet transform)
            x_reconstructed = low_pass_reconstructed + high_pass_reconstructed
        
        return x_reconstructed.squeeze(1)  # Remove extra channel dimension for the final output


class GaussianProcessReconstruction:
    def __init__(self, length_scale=1.0, amplitude=1.0):
        self.length_scale = length_scale
        self.amplitude = amplitude
        
    def rbf_kernel(self, x1, x2):
        """RBF (Gaussian) kernel"""
        dist = torch.cdist(x1.unsqueeze(-1), x2.unsqueeze(-1))
        return self.amplitude * torch.exp(-0.5 * dist ** 2 / self.length_scale ** 2)
    
    def fit_predict(self, x_train, y_train, x_pred):
        """Gaussian Process prediction"""
        K = self.rbf_kernel(x_train, x_train)
        K_star = self.rbf_kernel(x_train, x_pred)
        K_inv = torch.linalg.inv(K + 1e-6 * torch.eye(K.shape[0], device=K.device))
        
        mean = K_star.T @ K_inv @ y_train
        return mean

    def blend_signals(self, data1, data2, overlap_start, overlap_end):
        x = torch.arange(len(data1), device=data1.device, dtype=torch.float32)
        x_overlap = x[overlap_start:overlap_end]
        
        # Fit GP to both signals in overlap region
        pred1 = self.fit_predict(x[:overlap_end], data1[:overlap_end], x_overlap)
        pred2 = self.fit_predict(x[overlap_start:], data2, x_overlap)
        
        # Blend predictions based on uncertainty
        weight = torch.linspace(0, 1, len(x_overlap), device=data1.device)
        blended = pred1 * (1 - weight) + pred2 * weight
        
        return blended

class AttentionBasedReconstruction(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_proj = torch.nn.Linear(1, hidden_dim)
        self.key_proj = torch.nn.Linear(1, hidden_dim)
        self.value_proj = torch.nn.Linear(1, hidden_dim)
        self.output_proj = torch.nn.Linear(hidden_dim, 1)
        
    def forward(self, data1, data2, overlap_region):
        # Project signals to higher dimension
        q = self.query_proj(overlap_region.unsqueeze(-1))
        k1 = self.key_proj(data1.unsqueeze(-1))
        k2 = self.key_proj(data2.unsqueeze(-1))
        v1 = self.value_proj(data1.unsqueeze(-1))
        v2 = self.value_proj(data2.unsqueeze(-1))
        
        # Compute attention weights
        attn1 = F.softmax(torch.matmul(q, k1.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim)), dim=-1)
        attn2 = F.softmax(torch.matmul(q, k2.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim)), dim=-1)
        
        # Weighted combination
        out1 = torch.matmul(attn1, v1)
        out2 = torch.matmul(attn2, v2)
        
        # Progressive blending
        weight = torch.linspace(0, 1, len(overlap_region), device=data1.device).unsqueeze(-1)
        blended = self.output_proj(out1 * (1 - weight) + out2 * weight)
        
        return blended.squeeze(-1)

import torch

def unsplit_data_advanced(windowed_data, window_size, step_size, data_points_per_second, original_length):
    """Advanced reconstruction using multiple techniques"""
    device = "cuda"
    batch_size, num_windows, prediction_size = windowed_data.shape
    
    # Initialize reconstructors
    wavelet_reconstructor = WaveletBasedReconstruction()
    gp_reconstructor = GaussianProcessReconstruction()
    attention_reconstructor = AttentionBasedReconstruction().to(device)
    
    # Output container
    reconstructed = torch.zeros((batch_size, original_length), device=device)
    
    for b in range(batch_size):
        current_pos = 0
        prev_window = None
        
        for i in range(num_windows):
            current_window = windowed_data[b, i]
            
            if prev_window is not None:
                overlap_size = window_size - step_size
                overlap_start = current_pos
                overlap_end = current_pos + overlap_size
                
                # Get reconstructions from different methods
                wavelet_blend = wavelet_reconstructor.reconstruct_overlap(
                    prev_window, current_window, overlap_size)
                gp_blend = gp_reconstructor.blend_signals(
                    prev_window, current_window, overlap_start, overlap_end)
                attention_blend = attention_reconstructor(
                    prev_window, current_window, current_window[:overlap_size])
                
                # Combine reconstructions based on confidence/quality metrics
                confidence_wavelet = torch.exp(-torch.abs(torch.diff(wavelet_blend)))
                confidence_gp = torch.exp(-torch.abs(torch.diff(gp_blend)))
                confidence_attention = torch.exp(-torch.abs(torch.diff(attention_blend)))
                
                total_confidence = confidence_wavelet + confidence_gp + confidence_attention
                weights = torch.stack([
                    confidence_wavelet/total_confidence,
                    confidence_gp/total_confidence,
                    confidence_attention/total_confidence
                ])
                
                final_blend = (
                    wavelet_blend * weights[0] +
                    gp_blend * weights[1] +
                    attention_blend * weights[2]
                )
                
                reconstructed[b, overlap_start:overlap_end] = final_blend
            
            # Fill non-overlapping part
            non_overlap_start = current_pos + (overlap_size if prev_window is not None else 0)
            non_overlap_end = min(non_overlap_start + step_size, original_length)
            
            # Convert current_window to a torch tensor on the right device
            reconstructed[b, non_overlap_start:non_overlap_end] = torch.tensor(current_window, device=device)[
                (overlap_size if prev_window is not None else 0):
                (non_overlap_end - non_overlap_start + (overlap_size if prev_window is not None else 0))
            ]
            
            prev_window = current_window
            current_pos += step_size
    
    return reconstructed
