import os
import pandas as pd
import numpy as np
import scipy
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, train_test_split
from datetime import datetime
from utils_glenn import *  # Assuming this file contains necessary utility functions

def create_run_directory():
    base_dir = "runs"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def _calculate_flattened_accuracy(average, ground_truth_labels):
    s_acc = 0
    for b in range(len(ground_truth_labels)):
        s, _ = scipy.stats.pearsonr(average[b], ground_truth_labels[b])
        s_acc += s
    return s_acc / len(ground_truth_labels)

def _choose_real_labs_only_with_filenames(labels, filenames):
    return labels[labels['filename'].isin(filenames)]

def _get_ground_truth_labels(ground_truth_names, labels):
    ground_truth_labels = []
    for batch_name in ground_truth_names:
        ground_truth_label = _choose_real_labs_only_with_filenames(labels, [batch_name])
        ground_truth_labels.append(ground_truth_label)
    return np.array(ground_truth_labels)[:, :, -1].astype(np.float32)

class CustomTensorBoardCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, fold):
        super().__init__()
        self.log_dir = log_dir
        self.fold = fold
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, f"fold_{fold}"))

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(f"{name}_fold_{self.fold}", value, step=epoch)



def train(path_to_data, path_to_labels, window_size = 16, step_size =2/5, data_parts=4, epochs = 100, batch_size =10, early_stopping_patience = 20):
    run_dir = create_run_directory()
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Parameters
    length_sequence = window_size 
    step_sequence = int(step_size* window_size)

    # Load and prepare data
    train_data, train_labels, train_dict, frame_rate = load_data(path_to_data, path_to_labels, 'train')
    devel_data, devel_labels, devel_dict, frame_rate = load_data(path_to_data, path_to_labels, 'devel')
    test_data, test_labels, test_dict, frame_rate = load_data(path_to_data, path_to_labels, 'test')

    # Combine train and devel data
    all_data = np.concatenate((train_data, devel_data), axis=0)
    all_labels = pd.concat([train_labels, devel_labels])
    all_dict = np.concatenate((list(train_dict.values()), list(devel_dict.values())), axis=0)
    # Prepare data
    prepared_data, prepared_labels, prepared_labels_timesteps = prepare_data(all_data, all_labels, all_dict, frame_rate, length_sequence * 16000, step_sequence * 16000)
    prepared_test_data, prepared_test_labels, prepared_test_labels_timesteps = prepare_data(test_data, test_labels, test_dict, frame_rate, length_sequence * 16000, step_sequence * 16000)

    # Create CSV file for storing fold indices
    fold_indices_df = pd.DataFrame(columns=['Fold', 'Train_Indices', 'Val_Indices'])

    # Cross-validation
    kf = KFold(n_splits=data_parts)
    fold_metrics = []

    for fold, (train_index, val_index) in enumerate(kf.split(prepared_data)):
        print(f"Fold {fold + 1}/{data_parts}")

        # Save fold indices
        fold_indices_df = fold_indices_df._append({
            'Fold': fold + 1,
            'Train_Indices': train_index.tolist(),
            'Val_Indices': val_index.tolist()
        }, ignore_index=True)

        # Split data
        train_d, val_d = prepared_data[train_index], prepared_data[val_index]
        train_lbs, val_lbs = prepared_labels[train_index], prepared_labels[val_index]
        train_timesteps, val_timesteps = prepared_labels_timesteps[train_index], prepared_labels_timesteps[val_index]

        # Reshape data
        train_d, train_lbs = reshaping_data_for_model(train_d, train_lbs)
        val_d, val_lbs = reshaping_data_for_model(val_d, val_lbs)
        test_d, test_lbs = reshaping_data_for_model(prepared_test_data, prepared_test_labels)

        # Create and compile model
        model = create_1dcnn(input_shape=(train_d.shape[-2], train_d.shape[-1]))
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss=correlation_coefficient_loss,
                      metrics=['mse', 'mae', correlation_coefficient_accuracy])

        # Callbacks
        tb_callback = CustomTensorBoardCallback(log_dir, fold + 1)
        early_stopping = keras.callbacks.EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001)

        # Train the model
        history = model.fit(train_d, train_lbs, batch_size=batch_size, epochs=epochs,
                            validation_data=(val_d, val_lbs), 
                            callbacks=[tb_callback, early_stopping, reduce_lr])

        # Evaluate model on validation data
        
        val_pred = model.predict(val_d, batch_size=batch_size)
        val_pred = val_pred.reshape(val_timesteps.shape)
        val_ground_truth = _get_ground_truth_labels([all_dict[i] for i in val_index], all_labels)
        val_pred_flat = unsplit_data_ogsize(val_pred, window_size, step_sequence, 25, val_ground_truth.shape[-1])
        val_prc_coef = _calculate_flattened_accuracy(val_pred_flat, val_ground_truth)

        # Evaluate model on test data
        test_pred = model.predict(test_d, batch_size=batch_size)
        test_pred = test_pred.reshape(prepared_test_labels_timesteps.shape)
        test_ground_truth = _get_ground_truth_labels(list(test_dict.values()), test_labels)
        test_pred_flat = unsplit_data_ogsize(test_pred, window_size, step_sequence, 25, test_ground_truth.shape[-1])
        test_prc_coef = _calculate_flattened_accuracy(test_pred_flat, test_ground_truth)
        test_loss, test_mse, test_mae, test_acc = model.evaluate(test_d, test_lbs, batch_size=batch_size)

        print(f"Fold {fold + 1}:")
        print(f"  Validation Pearson Coefficient: {val_prc_coef}")
        print(f"  Test Loss: {test_loss}")
        print(f"  Test MSE: {test_mse}")
        print(f"  Test MAE: {test_mae}")
        print(f"  Test Pearson Coefficient: {test_acc}")
        print(f"  Test Pearson Coefficient (flattened): {test_prc_coef}")

        fold_metrics.append({
            'val_prc': val_prc_coef,
            'test_loss': test_loss,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_prc': test_acc,
            'test_prc_flat': test_prc_coef
        })

        # Save fold model
        model.save(os.path.join(run_dir, f'model_fold_{fold + 1}.h5'))

        # Log fold-specific metrics
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar(f"val_flat_pearson_coefficient_fold_{fold + 1}", val_prc_coef, step=1)
            tf.summary.scalar(f"test_loss_fold_{fold + 1}", test_loss, step=1)
            tf.summary.scalar(f"test_mse_fold_{fold + 1}", test_mse, step=1)
            tf.summary.scalar(f"test_mae_fold_{fold + 1}", test_mae, step=1)
            tf.summary.scalar(f"test_pearson_coefficient_fold_{fold + 1}", test_acc, step=1)
            tf.summary.scalar(f"test_pearson_coefficient_flat_fold_{fold + 1}", test_prc_coef, step=1)

    # Calculate and log average metrics
    avg_metrics = {key: np.mean([fold[key] for fold in fold_metrics]) for key in fold_metrics[0].keys()}
    
    with tf.summary.create_file_writer(log_dir).as_default():
        for key, value in avg_metrics.items():
            tf.summary.scalar(f"average_{key}", value, step=1)

    # Save the model with the best average performance
    best_fold = np.argmax([fold['test_prc_flat'] for fold in fold_metrics])
    best_model_path = os.path.join(run_dir, f'model_fold_{best_fold + 1}.h5')
    os.rename(best_model_path, os.path.join(run_dir, 'best_model.h5'))

    # Save fold indices CSV
    fold_indices_df.to_csv(os.path.join(run_dir, 'fold_indices.csv'), index=False)

    print("\nTraining completed.")
    print(f"Best model (Fold {best_fold + 1}) saved.")
    print("Average metrics across all folds:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    path = "/home/glenn/Downloads/"
    path  = "../DATA/"

    train(
        path_to_data=path+"ComParE2020_Breathing/wav/",
        path_to_labels=path+"ComParE2020_Breathing/lab/",
        window_size=16,
        batch_size=4
    )