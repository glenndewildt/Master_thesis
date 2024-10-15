### train.import os
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
from keras import Sequential
from keras.layers import Conv1D, MaxPool1D, LSTM, Dense, Dropout, Flatten, TimeDistributed, MultiHeadAttention, LayerNormalization
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
import librosa
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Layer, Reshape, Flatten, Concatenate
import soundfile as sf
# Load the smaller pretrained HuBERT base model and processor

import numpy as np
from keras.models import Sequential
from keras.layers import GRU, Dense
from tensorflow.keras import layers




def how_many_windows(total_length, window_size, step_size):
    return int(np.ceil((total_length - window_size) / step_size)) + 1




def reshaping_data_for_model(data, labels):
    result_data=data.reshape((-1,data.shape[2])+(1,))
    result_labels=labels.reshape((-1,labels.shape[2]))
    return result_data, result_labels


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

def prepare_data(data, labels, class_to_filename_dict, frame_rate, size_window, step_for_window):
    label_rate=25 # 25 Hz label rate
    num_windows=how_many_windows_do_i_need(data.shape[1],size_window, step_for_window)
    new_data=np.zeros(shape=(data.shape[0],int(num_windows),size_window))
    length_of_label_window=int(size_window/frame_rate*label_rate)
    step_of_label_window=int(length_of_label_window*(step_for_window/size_window))
    new_labels=np.zeros(shape=(np.unique(labels['filename']).shape[0], int(num_windows),length_of_label_window ))
    new_labels_timesteps=np.zeros(shape=new_labels.shape)
    for instance_idx in range(data.shape[0]):
        start_idx_data=0
        start_idx_label=0
        temp_labels=labels[labels['filename']==class_to_filename_dict[instance_idx]]
        temp_labels=temp_labels.drop(columns=['filename'])
        temp_labels=temp_labels.values
        for windows_idx in range(num_windows-1):
            new_data[instance_idx,windows_idx]=data[instance_idx,start_idx_data:start_idx_data+size_window]
            new_labels[instance_idx,windows_idx]=temp_labels[start_idx_label:start_idx_label+length_of_label_window, 1]
            new_labels_timesteps[instance_idx, windows_idx]=temp_labels[start_idx_label:start_idx_label+length_of_label_window, 0]
            start_idx_data+=step_for_window
            start_idx_label+=step_of_label_window
        if start_idx_data+size_window>=data.shape[1]:
            new_data[instance_idx,num_windows-1]=data[instance_idx, data.shape[1]-size_window:data.shape[1]]
            new_labels[instance_idx, num_windows-1]=temp_labels[temp_labels.shape[0]-length_of_label_window:temp_labels.shape[0],1]
            new_labels_timesteps[instance_idx, num_windows-1]=temp_labels[temp_labels.shape[0]-length_of_label_window:temp_labels.shape[0],0]
        else:
            new_data[instance_idx,num_windows-1]=data[instance_idx,start_idx_data:start_idx_data+size_window]
            new_labels[instance_idx,num_windows-1]=temp_labels[start_idx_label:start_idx_label+length_of_label_window, 1]
            new_labels_timesteps[instance_idx, num_windows-1]=temp_labels[start_idx_label:start_idx_label+length_of_label_window, 0]
            start_idx_data+=step_for_window
            start_idx_label+=step_of_label_window
    return new_data, new_labels, new_labels_timesteps




def instance_normalization(data):
    for instance_idx in range(data.shape[0]):
        scaler=StandardScaler()
        temp_data=data[instance_idx].reshape((-1,1))
        temp_data=scaler.fit_transform(temp_data)
        temp_data=temp_data.reshape((data.shape[1:]))
        data[instance_idx]=temp_data
    return data

def sample_standart_normalization(data, scaler=None):
    tmp_shape=data.shape
    tmp_data=data.reshape((-1,1))
    if scaler==None:
        scaler=StandardScaler()
        tmp_data=scaler.fit_transform(tmp_data)
    else:
        tmp_data=scaler.transform(tmp_data)
    data=tmp_data.reshape(tmp_shape)
    return data

def sample_minmax_normalization(data, min=None, max=None):
    result_shape=data.shape
    tmp_data=data.reshape((-1))
    if max==None:
        max=np.max(tmp_data)
    if min == None:
        min=np.min(tmp_data)
    tmp_data=2*(tmp_data-min)/(max-min)-1
    data=tmp_data.reshape(result_shape)
    return data, min, max



def create_1dcnn(input_shape):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(input_shape=input_shape, filters=64, kernel_size=10, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=10))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=8, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=6, strides=1, activation='relu', padding='same'       ))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, activation='relu', padding='same'       ))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.AvgPool1D(pool_size=4))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='tanh')))
    model.add(tf.keras.layers.Flatten())
    print(model.summary())
    return model



def correlation_coefficient_accuracy(y_true, y_pred):
    #squeezed_tensor = tf.squeeze(y_true, axis=-1)

    x = y_true
    y = y_pred
    mx = K.mean(x, axis=1, keepdims=True)
    my = K.mean(y, axis=1, keepdims=True)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym), axis=1)
    sum_square_x = K.sum(K.square(xm), axis=1)
    sum_square_y = K.sum(K.square(ym), axis=1)
    sqrt_x = tf.sqrt(sum_square_x)
    sqrt_y = tf.sqrt(sum_square_y)
    r_den = tf.multiply(sqrt_x, sqrt_y)
    r = tf.divide(r_num, r_den)
    # To avoid NaN in division, we handle the case when r_den is 0
    r = tf.where(tf.math.is_nan(r), tf.zeros_like(r), r)
    return K.mean(r)



def create_model_parems(input_shape, filters_list, kernel_size_list, dropout_rate_list, pool_size_list, lstm_units_list, activation_list, final_activation):
    model = tf.keras.Sequential()
    
    # Add Conv1D and pooling layers based on the provided lists
    
    for i in range(len(filters_list)):
        if i == 0:
            model.add(tf.keras.layers.Conv1D(filters=filters_list[i], kernel_size=kernel_size_list[i], strides=1,
                                             activation=activation_list[i], padding='same', input_shape=input_shape ))
            model.add(tf.keras.layers.Dropout(dropout_rate_list[i]))
            model.add(tf.keras.layers.MaxPool1D(pool_size=pool_size_list[i]))
        else:
            model.add(tf.keras.layers.Conv1D(filters=filters_list[i], kernel_size=kernel_size_list[i], strides=1,
                                             activation=activation_list[i], padding='same'))
            model.add(tf.keras.layers.Dropout(dropout_rate_list[i]))
            model.add(tf.keras.layers.MaxPool1D(pool_size=pool_size_list[i]))
    
    # Add LSTM layers
    for units in lstm_units_list:
        model.add(tf.keras.layers.LSTM(units, return_sequences=True))
    
    # Add TimeDistributed Dense layer
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation=final_activation)))
    
    # Flatten the output
    x = model.add(tf.keras.layers.Flatten())
    print(model.summary())
    
    return model


def correlation_coefficient_loss(y_true, y_pred):
    x=y_true
    y=y_pred
    mx=K.mean(x, axis=1, keepdims=True)
    my=K.mean(y, axis=1, keepdims=True)
    xm,ym=x-mx,y-my
    r_num=K.sum(tf.multiply(xm, ym), axis=1)
    sum_square_x=K.sum(K.square(xm), axis=1)
    sum_square_y = K.sum(K.square(ym), axis=1)
    sqrt_x = tf.sqrt(sum_square_x)
    sqrt_y = tf.sqrt(sum_square_y)
    r_den=tf.multiply(sqrt_x, sqrt_y)
    result=tf.divide(r_num, r_den)
    #tf.print('result:', result)
    result=K.mean(result)
    #tf.print('mean result:', result)
    return 1 - result

def pearson_coef(y_true, y_pred):
    return scipy.stats.pearsonr(y_true, y_pred)

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
    original_data = np.where(overlap_count != 0, original_data / overlap_count, np.zeros_like(original_data))

    # Trim the data to match the original length
    original_data = original_data[:, :original_length]

    return original_data

def concatenate_prediction(predicted_values, labels_timesteps, filenames_dict, columns_for_real_labels=['filename', 'timeFrame', 'upper_belt']):
    predicted_values = predicted_values.reshape(labels_timesteps.shape)
    result_predicted_values = pd.DataFrame(columns=columns_for_real_labels, dtype='float32')
    result_predicted_values['filename'] = result_predicted_values['filename'].astype('str')
    for instance_idx in range(0, predicted_values.shape[0]):
        predicted_values_tmp = predicted_values[instance_idx].reshape((-1, 1))
        timesteps_labels_tmp = labels_timesteps[instance_idx].reshape((-1, 1))
        tmp = pd.DataFrame(columns=['timeFrame', 'upper_belt'],
                           data=np.concatenate((timesteps_labels_tmp, predicted_values_tmp), axis=1))
        tmp = tmp.groupby(by=['timeFrame']).mean().reset_index()
        tmp['filename'] = filenames_dict[instance_idx]
        result_predicted_values = pd.concat([result_predicted_values,tmp.copy(deep=True)], ignore_index=True)
    result_predicted_values['timeFrame'] = result_predicted_values['timeFrame'].astype('float32')
    result_predicted_values['upper_belt'] = result_predicted_values['upper_belt'].astype('float32')
    return result_predicted_values[columns_for_real_labels]
