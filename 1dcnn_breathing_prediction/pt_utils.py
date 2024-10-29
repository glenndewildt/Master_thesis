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
class KalmanFilter:
    def __init__(self, initial_state_mean, state_covariance, observation_covariance):
        self.initial_state_mean = initial_state_mean
        self.state_covariance = state_covariance
        self.observation_covariance = observation_covariance
        self.current_state_estimate = initial_state_mean

    def predict(self):
        # Prediction step (state transition model)
        self.current_state_estimate = self.current_state_estimate
        self.state_covariance = self.state_covariance + self.observation_covariance

    def update(self, observation):
        # Update step
        kalman_gain = self.state_covariance / (self.state_covariance + self.observation_covariance)
        self.current_state_estimate += kalman_gain * (observation - self.current_state_estimate)
        self.state_covariance *= (1 - kalman_gain)

    def filter(self, observations):
        estimates = []
        for observation in observations:
            self.predict()
            self.update(observation)
            estimates.append(self.current_state_estimate.clone())
        return torch.stack(estimates)


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
    predicted_values = predicted_values.reshape(timesteps_labels.shape)
    tmp = np.zeros(shape=(true_values.shape[0], 3))
    result_predicted_values = pd.DataFrame(data=tmp, columns=true_values.columns, dtype='float32')
    result_predicted_values['filename'] = result_predicted_values['filename'].astype('str')

    index_temp = 0
    for instance_idx in range(0,predicted_values.shape[0]-1):
        # Round the timesteps to 5 decimal places when getting unique values
        timesteps = np.unique(np.round(timesteps_labels[instance_idx], decimals=5))
        
        for timestep in timesteps:
            # assignment for filename and timestep
            result_predicted_values.iloc[index_temp, 0] = class_dict[instance_idx]
            # Store the rounded timestep
            result_predicted_values.iloc[index_temp, 1] = round(timestep, 5)
            
            # Calculate mean of windows using rounded comparison
            mask = np.round(timesteps_labels[instance_idx], decimals=5) == timestep
            result_predicted_values.iloc[index_temp, 2] = np.mean(predicted_values[instance_idx, mask])
            index_temp += 1
            
    result = result_predicted_values.iloc[:, 2].to_numpy()    
    result = result.reshape(predicted_values.shape[0],-1)
    return result

def unsplit_data(windowed_data,window_size, step_size, method, original_length, data_points_per_second= 25):

    if method == 'original':
        return unsplit_data_ogsize(windowed_data, window_size, step_size, data_points_per_second, original_length)
    elif method == 'gaussian':
        return unsplit_data_gaussian(windowed_data, window_size, step_size, data_points_per_second, original_length)
    elif method == 'cubic':
        return unsplit_data_cubic(windowed_data, window_size, step_size, data_points_per_second, original_length)
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
        original_data = torch.where(overlap_count != 0, original_data / overlap_count, torch.zeros_like(original_data))

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

def unsplit_data_kalman(windowed_data, window_size, step_size, data_points_per_second, original_length):
    # Ensure windowed_data is a tensor on the GPU
    windowed_data = torch.tensor(windowed_data, device='cuda')

    batch_size, num_windows, prediction_size = windowed_data.shape
    window_size_points = int(window_size * data_points_per_second)
    step_size_points = int(step_size * data_points_per_second)

    # Initialize tensors on GPU
    original_data = torch.zeros((batch_size, original_length), device='cuda')
    overlap_count = torch.zeros((batch_size, original_length), device='cuda')

    # Accumulate data using windowed_data
    for i in range(num_windows):
        start = i * step_size_points
        end = min(start + window_size_points, original_length)
        segment_length = end - start

        # Use slicing to accumulate results across all batches
        original_data[:, start:end] += windowed_data[:, i, :segment_length]
        overlap_count[:, start:end] += 1

    # Average the overlapping regions
    original_data = torch.where(overlap_count != 0, original_data / overlap_count, torch.zeros_like(original_data))

    # Kalman Filter parameters
    initial_state_mean = 0.0  # Adjust as needed
    state_covariance = 0.5  # Decreased to reflect lower uncertainty
    observation_covariance = 0.1  # Adjusted to reflect observation noise

    # Apply Kalman Filtering to the entire signal
    kalman_filter = KalmanFilter(initial_state_mean, state_covariance, observation_covariance)
    for b in range(batch_size):
        original_data[b] = kalman_filter.filter(original_data[b])

    return original_data.cpu()