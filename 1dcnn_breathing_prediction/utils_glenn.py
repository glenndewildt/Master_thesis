### train.import os
import pandas as pd
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
import librosa
import tensorflow as tf
from tensorflow import keras
from keras.saving import register_keras_serializable

from keras import saving
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-Head Self Attention
    attention_output = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    # Add & Norm
    attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        inputs + attention_output
    )
    # Feedforward Network
    ff_output = tf.keras.layers.Dense(ff_dim, activation="relu")(attention_output)
    ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
    ff_output = tf.keras.layers.Dense(inputs.shape[-1])(ff_output)
    # Add & Norm
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)

def create_1dcnn_with_transformer(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Conv1D Layers
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=10, strides=1, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.MaxPool1D(pool_size=10)(x)
    
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=8, strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=6, strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.AvgPool1D(pool_size=4)(x)

    
    # Transformer Encoder Layer
    x = transformer_encoder(x, head_size=32, num_heads=16, ff_dim=512, dropout=0.1)
    x = transformer_encoder(x, head_size=32, num_heads=16, ff_dim=512, dropout=0.1)

    print(x.shape)
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same')(x)
    print(x.shape)
    x= tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(x)
    x= tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(x)

    print(x.shape)
    # Output layer for sequence regression (400 time steps)
    outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
    outputs = tf.keras.layers.Flatten()(outputs)

    print(outputs.shape)

    # Build the final model
    model = tf.keras.Model(inputs, outputs)
    print(model.summary())

    return model


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
    model.add(tf.keras.layers.Conv1D(input_shape=input_shape, filters=256, kernel_size=10, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=10))
    model.add(tf.keras.layers.Conv1D(filters=512, kernel_size=8, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', padding='same'       ))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(filters=1024, kernel_size=5, strides=1, activation='relu', padding='same'       ))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.AvgPool1D(pool_size=4))
    model.add(tf.keras.layers.LSTM(1024, return_sequences=True))
    model.add(tf.keras.layers.LSTM(1024, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='tanh')))
    model.add(tf.keras.layers.Flatten())
    print(model.summary())
    return model

def create_1dcnn_lstm_arch(input_shape):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(input_shape=input_shape, filters=64, kernel_size=10, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=10))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=8, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(filters=258, kernel_size=6, strides=1, activation='relu', padding='same'       ))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(filters=258, kernel_size=5, strides=1, activation='relu', padding='same'       ))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.AvgPool1D(pool_size=4))
    model.add(tf.keras.layers.LSTM(258, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.LSTM(258, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.LSTM(258, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='tanh')))
    model.add(tf.keras.layers.Flatten())
    print(model.summary())
    return model

def create_1dcnn_lstm_cnn_arch(input_shape):
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
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, activation='relu', padding='same'       ))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='tanh')))
    model.add(tf.keras.layers.Flatten())
    print(model.summary())
    return model

def create_1dcnn_bilstm_cnn_arch(input_shape):
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
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, activation='relu', padding='same'       ))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='tanh')))
    model.add(tf.keras.layers.Flatten())
    print(model.summary())
    return model

def create_1dcnn_arch(input_shape):
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
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, activation='relu', padding='same'       ))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='tanh')))
    model.add(tf.keras.layers.Flatten())
    print(model.summary())
    return model

def create_cnn_bilstm_arch(input_shape):
    model = tf.keras.Sequential([
        # CNN layers
        tf.keras.layers.Conv1D(input_shape=input_shape, filters=64, kernel_size=10, strides=1, activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPool1D(pool_size=10),
        
        tf.keras.layers.Conv1D(filters=128, kernel_size=8, strides=1, activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPool1D(pool_size=4),
        
        tf.keras.layers.Conv1D(filters=256, kernel_size=6, strides=1, activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPool1D(pool_size=4),
        
        tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.AvgPool1D(pool_size=4),
        
        tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.3),
        
        # BiLSTM layers
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        
        # Output layer
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='tanh')),
        tf.keras.layers.Flatten()
    ])
    
    model.summary()
    return model

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

@register_keras_serializable()
def correlation_coefficient_accuracy(y_true, y_pred):
    # Mean of true and predicted values
    mx = tf.reduce_mean(y_true, axis=1, keepdims=True)
    my = tf.reduce_mean(y_pred, axis=1, keepdims=True)

    # Centered values
    xm = y_true - mx
    ym = y_pred - my

    # Numerator of the correlation coefficient
    r_num = tf.reduce_sum(xm * ym, axis=1)

    # Denominator of the correlation coefficient
    sum_square_x = tf.reduce_sum(tf.square(xm), axis=1)
    sum_square_y = tf.reduce_sum(tf.square(ym), axis=1)
    
    sqrt_x = tf.sqrt(sum_square_x)
    sqrt_y = tf.sqrt(sum_square_y)

    r_den = sqrt_x * sqrt_y

    # Calculate correlation coefficient
    result = r_num / r_den
    
    # Mean of the correlation coefficient
    result = tf.reduce_mean(result)

    return result

@register_keras_serializable()
def correlation_coefficient_loss(y_true, y_pred):
    # Mean of true and predicted values
    mx = tf.reduce_mean(y_true, axis=1, keepdims=True)
    my = tf.reduce_mean(y_pred, axis=1, keepdims=True)

    # Centered values
    xm = y_true - mx
    ym = y_pred - my

    # Numerator of the correlation coefficient
    r_num = tf.reduce_sum(xm * ym, axis=1)

    # Denominator of the correlation coefficient
    sum_square_x = tf.reduce_sum(tf.square(xm), axis=1)
    sum_square_y = tf.reduce_sum(tf.square(ym), axis=1)
    
    sqrt_x = tf.sqrt(sum_square_x)
    sqrt_y = tf.sqrt(sum_square_y)

    r_den = sqrt_x * sqrt_y

    # Calculate correlation coefficient
    result = r_num / r_den
    
    # Mean of the correlation coefficient
    result = tf.reduce_mean(result)

    # Return 1 - correlation coefficient
    return 1 - result

@register_keras_serializable()
def concordance_correlation_coefficient_loss(y_true, y_pred):
    # Mean of true and predicted values
    mean_true = tf.reduce_mean(y_true, axis=1, keepdims=True)
    mean_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)
    
    # Variance of true and predicted values
    var_true = tf.reduce_mean(tf.square(y_true - mean_true), axis=1) + tf.keras.backend.epsilon()  # Add epsilon to prevent div by zero
    var_pred = tf.reduce_mean(tf.square(y_pred - mean_pred), axis=1) + tf.keras.backend.epsilon()
    
    # Covariance between true and predicted values
    covariance = tf.reduce_mean((y_true - mean_true) * (y_pred - mean_pred), axis=1)
    
    # CCC numerator: 2 * covariance
    ccc_num = 2 * covariance
    
    # CCC denominator: var_true + var_pred + (mean_true - mean_pred)^2
    ccc_den = var_true + var_pred + tf.square(mean_true - mean_pred)
    
    # CCC calculation
    ccc = ccc_num / ccc_den
    
    # Handle NaN values (in case of zero denominator)
    ccc = tf.where(tf.math.is_nan(ccc), tf.zeros_like(ccc), ccc)
    
    # Loss is 1 - CCC
    ccc_loss = 1 - tf.reduce_mean(ccc)
    
    return ccc_loss



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
