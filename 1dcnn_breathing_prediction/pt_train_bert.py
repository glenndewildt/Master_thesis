import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from datetime import datetime
#from torch.utils.tensorboard import SummaryWriter
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
from typing import List, Tuple
from pt_utils import *
from pt_dataset import *
from pt_models import *
from pt_utils import *
from tensorboardX import SummaryWriter
from transformers import get_cosine_schedule_with_warmup


def create_run_directory():
    base_dir = "pt_runs"
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

def train(path_to_data, path_to_labels, window_size=16, step_size=6, data_parts=4, epochs=100, batch_size=10, early_stopping_patience=20, config = None, model =None, processor = None):
    run_dir = create_run_directory()
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Parameters
    length_sequence = window_size 
    step_sequence = step_size

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
    prepared_test_data, prepared_test_labels, prepared_test_labels_timesteps = prepare_data(test_data, test_labels, test_dict, frame_rate, length_sequence * 16000, 1 * 16000)

    # Create CSV file for storing fold indices
    fold_indices_df = pd.DataFrame(columns=['Fold', 'Train_Indices', 'Val_Indices'])

    # Cross-validation
    kf = KFold(n_splits=data_parts)
    fold_metrics = []
    # To accumulate metrics across folds for each epoch
    train_acc_epoch = []
    val_acc_epoch = []
    test_acc_epoch = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    config["output_size"] = prepared_labels.shape[-1]
    writer = SummaryWriter(log_dir=os.path.join(log_dir,config["model_name"]))

    for fold, (train_index, val_index) in enumerate(kf.split(prepared_data)):
        print(f"Fold {fold + 1}/{data_parts}")
        best_model_path = f"{run_dir}/best_model_fold{fold+1}"
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
        
        print(train_d.shape)

        # Create datasets
        train_dataset = BreathingDataset(train_d, train_lbs, processor, window_size, step_sequence)
        #train_dataset = GPUBreathingDataset(train_d, train_lbs, processor, augment=True)
        val_dataset = BreathingDataset(val_d, val_lbs, processor, window_size, step_sequence)
        test_dataset = BreathingDataset(test_d, test_lbs, processor, window_size, step_sequence)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=5, collate_fn=val_dataset.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5, collate_fn=test_dataset.collate_fn)
        print(config["output_size"])
        # Create and initialize model
        model = config["model"](config).to(device)
        ## uses scadular and optimiser from the parems
        ## training optimiser parameters
        learning_rate = 5e-4
        weight_decay = 0.001
        # Optimizer: AdamW
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Define total steps and warmup steps
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * 0.1)
        # Scheduler: CosineAnnealingWarmRestarts
        t0 = 10  # Number of epochs before the first restart
        t_mult = 2  # Factor to increase T_i after restart
        min_lr = 1e-5  # Minimum learning rate
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                             num_warmup_steps=warmup_steps, 
                                             num_training_steps=total_steps)

        #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=t_mult, eta_min=min_lr)
        # TensorBoard writer

        best_val_loss = float('inf')
        early_stopping_counter = 0
        # To accumulate metrics across folds for each epoch
        train_acc = []
        val_acc = []
        test_acc = []
        for epoch in range(epochs):

            model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for input_values, batch_lbs in progress_bar:
                optimizer.zero_grad()
                
                input_values = input_values.to(device)
                batch_lbs = batch_lbs.to(device)
                
                outputs = model(input_values)
                loss = correlation_coefficient_loss(outputs, batch_lbs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                progress_bar.set_postfix({'train_loss': f'{train_loss/(progress_bar.n+1):.4f}'})
                scheduler.step()


            train_loss /= len(train_loader)

            # Combined validation loop
            model.eval()
            val_loss = 0.0
            val_pred = []
            with torch.no_grad():
                for batch_d, batch_lbs in val_loader:
                    input_values = batch_d.to(device)
                    batch_lbs = batch_lbs.to(device)
                    
                    outputs = model(input_values)
                    loss = correlation_coefficient_loss(outputs, batch_lbs)
                    val_loss += loss.item()
                    val_pred.extend(outputs.cpu().numpy())

            val_loss /= len(val_loader)

            # Calculate validation metrics
            val_pred = np.array(val_pred).reshape(val_timesteps.shape)
            val_ground_truth = _get_ground_truth_labels([all_dict[i] for i in val_index], all_labels)
            val_pred_flat = unsplit_data_ogsize(val_pred, window_size, step_sequence, 25, val_ground_truth.shape[-1])
            val_prc_coef = _calculate_flattened_accuracy(val_pred_flat, val_ground_truth)
            
            # Accumulate metrics for this fold and epoch
            train_acc.append(1- train_loss)
            val_acc.append(1- val_loss)

            # Log metrics
            writer.add_scalar(f"Loss/train_fold_{fold + 1}", train_loss, epoch)
            writer.add_scalar(f"Loss/val_fold_{fold + 1}", val_loss, epoch)
            writer.add_scalar(f"Pearson/val_fold_{fold + 1}", val_prc_coef, epoch)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Pearson: {val_prc_coef:.4f}")

            # Check if validation loss improved
            if val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving best model...")
                best_val_loss = val_loss
                early_stopping_counter = 0

                # Save the best model
                torch.save(model.state_dict(), best_model_path)
            else:
                early_stopping_counter += 1
                print(f"Validation loss did not improve for {early_stopping_counter} epochs.")
                #model.load_state_dict(torch.load(best_model_path))


            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Loading best model.")
                # Load the best model's weights
                model.load_state_dict(torch.load(best_model_path))
                break

            # Adjust the learning rate based on validation loss

            #scheduler.step(val_loss)


        # Evaluate model on test data
        test_pred = []
        test_loss = 0.0
        with torch.no_grad():
            for batch_d, batch_lbs in test_loader:
                input_values = batch_d.to(device)
                batch_lbs = batch_lbs.to(device)
                
                outputs = model(input_values)
                loss = correlation_coefficient_loss(outputs, batch_lbs)
                test_loss += loss.item()
                test_pred.extend(outputs.cpu().numpy())

        test_loss /= len(test_loader)
        test_pred = np.array(test_pred).reshape(prepared_test_labels_timesteps.shape)
        test_ground_truth = _get_ground_truth_labels(list(test_dict.values()), test_labels)
        test_pred_flat = unsplit_data_ogsize(test_pred, window_size, step_sequence, 25, test_ground_truth.shape[-1])
        test_prc_coef = _calculate_flattened_accuracy(test_pred_flat, test_ground_truth)

        print(f"Fold {fold + 1}:")
        print(f"  Validation Pearson Coefficient  acc: {1- val_loss}")
        print(f"  Validation Pearson Coefficient flat acc: {val_prc_coef}")
        print(f"  Test acc: {1- test_loss}")
        print(f"  Test Pearson Coefficient acc(flattened): {test_prc_coef}")

        fold_metrics.append({
            'Fold': fold + 1,
            'val_prc_acc': 1- val_loss,
            'val_prc_acc_flat': val_prc_coef,
            'test_acc': 1- test_loss,
            'test_prc_flat': test_prc_coef
        })


                # Log fold-specific metrics as tables
        fold_table = f"| Fold | Val Pearson Acc | Val Pearson Flat | Test Acc | Test Pearson Flat |\n" \
                     f"|------|-----------------|------------------|----------|-------------------|\n" \
                     f"| {fold + 1} | {1 - val_loss:.4f} | {val_prc_coef:.4f} | {1 - test_loss:.4f} | {test_prc_coef:.4f} |\n"
        writer.add_text(f"Fold_{fold + 1}_Metrics", fold_table)
        # Accumulate fold metrics across all folds
        train_acc_epoch.append(train_acc)
        val_acc_epoch.append(train_acc)


    
        # After all folds, compute and log the average metrics per epoch across all folds
    for epoch in range(epochs):
        avg_train_loss = np.mean([fold_losses[epoch] for fold_losses in train_acc_epoch if len(fold_losses) > epoch])
        avg_val_loss = np.mean([fold_losses[epoch] for fold_losses in val_acc_epoch if len(fold_losses) > epoch])

        # Log the averaged metrics for the epoch across all folds
        writer.add_scalar("Average_acc/train", avg_train_loss, epoch)
        writer.add_scalar("Average_acc/val", avg_val_loss, epoch)
            

    # Calculate average metrics
    avg_metrics = {key: np.mean([fold[key] for fold in fold_metrics if key != 'Fold']) for key in fold_metrics[0].keys() if key != 'Fold'}
        # Log the final average table
    avg_table = "| Fold | Val Pearson Acc | Val Pearson Flat | Test Acc | Test Pearson Flat |\n" \
                "|------|-----------------|------------------|----------|-------------------|\n" \
                f"| Average | {avg_metrics['val_prc_acc']:.4f} | {avg_metrics['val_prc_acc_flat']:.4f} | {avg_metrics['test_acc']:.4f} | {avg_metrics['test_prc_flat']:.4f} |\n"
    writer.add_text("Average_Metrics", avg_table)
    # Add average metrics to results
    avg_metrics['Fold'] = 'Average'
    fold_metrics.append(avg_metrics)

    # save averga date to CSV
    results_df = pd.DataFrame(fold_metrics)
    csv_path = os.path.join(run_dir, 'fold_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    # Save fold indices CSV
    fold_indices_df.to_csv(os.path.join(run_dir, 'fold_indices.csv'), index=False)
    
    writer.close()


    print("\nTraining completed.")
    print("Average metrics across all folds:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    ## Path to data
    path = "/home/glenn/Downloads/"
    path = "../DATA/"


    # Model parameters
    model_config = {
        "VRBModel": {
            "model" : VRBModel,
            "model_name": "facebook/hubert-large-ls960-ft",
            "hidden_units": 64,
            "n_gru": 3,
            "output_size": None  # Will be set dynamically
        },
        "Wav2Vec2ConvLSTMModel": {
            "model" : Wav2Vec2ConvLSTMModel,
            "model_name": "facebook/wav2vec2-base",
            "hidden_units": 128,
            "n_lstm": 2,
            "output_size": None  # Will be set dynamically
        },
        "RespBertLSTMModel": {
            'model': RespBertLSTMModel,
            "model_name": "microsoft/wavlm-large",
            "hidden_units": 256,
            "n_lstm": 2,
            "output_size": None  
        },
        "RespBertAttionModel": {
            'model' : RespBertAttionModel,
            "model_name": "microsoft/wavlm-large",
            "hidden_units": 512,
            "n_attion": 2,
            "output_size": None  
        },
            "RespBertCNNModel": {
            'model' : RespBertCNNModel,
            "model_name": "microsoft/wavlm-large",
            "hidden_units": 256,
            "output_size": None  
        },"VBR_WALM":{
            "model" : VRBModel,
            "model_name": "microsoft/wavlm-large",
            "hidden_units": 128,
            "n_gru": 3,
            "output_size": None  # Will be set dynamically
        },
    }
    

    
    # Train and data parameters
    epochs = 70
    batch_size = 20
    window_size = 30
    step_size = 6
    data_parts = 4 # aka folds
    early_stopping_patience = 10
    
    config = model_config["VBR_WALM"]
    #model
    
    model = None

    #processor = AutoProcessor.from_pretrained(config["model_name"])
    processor = Wav2Vec2FeatureExtractor.from_pretrained(config["model_name"])


    train(
        path_to_data=path+"ComParE2020_Breathing/wav/normalized/",
        path_to_labels=path+"ComParE2020_Breathing/lab/normalized/",
        window_size=window_size,
        batch_size=batch_size,
        config = config,
        step_size=step_size,
        data_parts= data_parts ,
        early_stopping_patience= early_stopping_patience,
        epochs= epochs,
        model= model,
        processor = processor
    )