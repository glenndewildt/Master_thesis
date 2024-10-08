import os
import datetime
import numpy as np
import csv
import scipy
import torch
import torch.optim as optim

from tqdm import tqdm
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset, random_split, SubsetRandomSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from transformers import Wav2Vec2ForCTC,  Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Config, WavLMModel, WavLMConfig, HubertModel, HubertConfig, HubertPreTrainedModel
# Local imports
from models import *
from utils import *
from losses import *
from dataset import *
from config import Config
#from torchviz import make_dot
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from scipy.stats import norm
from scipy.optimize import minimize
from pykalman import KalmanFilter
import concurrent.futures
from scipy.signal import get_window
from scipy.ndimage import label
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.signal import butter, sosfilt
from scipy import signal
from torch.cuda.amp import GradScaler, autocast
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

class Trainer:
    def __init__(self, config, model_classes, criterion, device, bert_config, ground_labels, processor):
        self.scaler = GradScaler()

        self.ground_labels = ground_labels
        self.config = config
        self.model_classes = model_classes
        self.criterion = criterion
        self.device = device
        self.bert_config = bert_config
        self.run_dir = self._create_run_directory()
        self.processor = processor
        self.csv_file = os.path.join(self.run_dir, "results_summary.csv")
        self._create_csv_file()
        
    def _log_to_csv(self, model_name, fold, best_val_loss, test_loss, test_acc):
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_name, fold, best_val_loss, test_loss,test_acc])
            
    def _log_data_used_to_csv(self, model_name, fold, train_data, val_data):
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Model', 'Fold', 'Index_train_data', ' Index_validation_data', 'len_validation'])
            writer.writerow([model_name, fold, train_data,val_data, len(val_data)])
            writer.writerow(['Model', 'Fold', 'Best Val Loss', 'Test Loss', 'Test Acc'])


    def _create_run_directory(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.config.log_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _create_csv_file(self):
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            #writer.writerow(['Model', 'Fold', 'Best Val Loss', 'Test Loss', 'Test Acc'])

    def train(self, train_data, test_data):
        #kfold = KFold(n_splits=self.config.n_folds, shuffle=True, random_state= 42)
        kfold = KFold(n_splits=self.config.n_folds)
        self.criterion = self.criterion.to("cuda")

        for model_name, model_class in self.model_classes.items():
            print(f"Training {model_name}...")
            model_results = []
            
            writer = SummaryWriter(os.path.join(self.run_dir, f"{model_name}"))
            self.writer = writer
            for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
                if fold < (self.config.n_folds -1):
                    continue

                print(f"Fold {fold + 1}/{self.config.n_folds}")

                train_sampler = SubsetRandomSampler(train_idx)
                val_sampler = SubsetRandomSampler(val_idx)
                self._log_data_used_to_csv(model_name, fold, train_idx, val_idx)
                #val_loader = DataLoader(train_data, batch_size=self.config.batch_size, sampler=val_sampler)

                train_loader = DataLoader(train_data, batch_size=self.config.batch_size, sampler=train_sampler,num_workers=5)
                val_loader = DataLoader(train_data, batch_size=self.config.batch_size, sampler=val_sampler,num_workers=5)
                test_loader = DataLoader(test_data, batch_size=(self.config.batch_size*2), num_workers=5)

                model_config = self.config.models[model_name]
                model_config['output_size'] = train_data.get_output_shape()

                model = model_class(bert_config=self.bert_config, config=model_config)
                model = model.to(self.device)
                #Load from path
                #/home/gdwildt/Master_thesis/breathing_prediction/results/logs/attaion_model_20_epochs_85/RespBertAttionModel_best_model_fold_6.pt
                #/home/gdwildt/Master_thesis/breathing_prediction/results/logs/attaion_model_20_epochs_85
                #/home/gdwildt/Master_thesis/breathing_prediction/results/logs/Attention_Waml_20_epochs_folds5/RespBertAttionModel_best_model_fold_0.pt
                #breathing_prediction/results/logs/bi_lstm_0_7_no_adap/RespBertLSTMModel_best_model_fold_19.pt
                #breathing_prediction/results/logs/LSTM_85_0-10/RespBertLSTMModel_best_model_fold_6.pt
                #breathing_prediction/results/logs/bi_lstm_0-7/RespBertLSTMModel_best_model_fold_6.pt
                #model.load_state_dict(torch.load("../results/logs/bi_lstm_0-7V2/RespBertLSTMModel_best_model_fold_6.pt", map_location=self.device))

                optimizer = AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.config.t0, T_mult=self.config.t_mult, eta_min=self.config.min_lr)
                
                best_val_loss = float('inf')
                best_model_path = None
                early_stopping = EarlyStopping(patience=self.config.patience, mode='min')
                for epoch in range(self.config.epochs):

                    #val_loss, val_acc , val_flat_acc, val_flat_acc_ola = self._evaluate(model, val_loader)

                    train_loss, train_acc = self._train_epoch(model, train_loader, optimizer,scheduler, epoch, self.config.epochs)
                    val_loss, val_acc , val_flat_acc, val_flat_acc_ola = self._evaluate(model, val_loader)
                    test_loss, test_acc, test_flat_acc,  test_flat_acc_ola = self._evaluate(model, test_loader)

                    self._log_metrics(writer, epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, fold,test_flat_acc, val_flat_acc, test_flat_acc_ola, val_flat_acc_ola)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if best_model_path:
                            os.remove(best_model_path)
                        best_model_path = os.path.join(self.run_dir, f"{model_name}_best_model_fold_{fold}.pt")
                        torch.save(model.state_dict(), best_model_path)
                    

                                # Early stopping
                    if early_stopping.step(val_loss):
                        print(f"Early stopping triggered for {model_name} at epoch {epoch+1}")
                        break
                
                model_results.append({
                    'fold': fold,
                    'best_val_loss': best_val_loss,
                    'test_loss': test_loss,
                    'test_acc': test_acc
                })
                
                self._log_to_csv(model_name, fold, best_val_loss, test_loss, test_acc)
            
            avg_results = self._calculate_average_results(model_results)
            self._log_average_results(writer, avg_results)
            self._log_to_csv(model_name, 'Average', avg_results['best_val_loss'], avg_results['test_loss'], avg_results['test_acc'])
            
            writer.close()
            self._print_model_results(model_name, model_results, avg_results)

    def _train_epoch(self, model, dataloader, optimizer,scheduler, epoch, total_epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        
        input_values, labels = flatten_and_shuffle_data(dataloader)
        train_dataset = AugmentedDataset(input_values, labels)
        dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=12)   
        l = len(dataloader.dataset)
            
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{total_epochs}")
        i = 0
        for batch_idx, (input_values, labels) in enumerate(progress_bar):
                        # Wait for the GPU to finish all operations before starting a new iteration
            torch.cuda.empty_cache()  # Clear cache to free up memory
            torch.cuda.synchronize()   # Ensure all GPU operations are completed

            s = input_values.shape

            input_values = np.stack(input_values, axis= 0)
                        #input_values = input_values.reshape(input_values.shape[1], input_values.shape[-1])


            #input_values, labels = input_values.to(self.device), labels.to(self.device) 
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                input_values = self.processor(input_values, return_tensors="pt", padding="longest", sampling_rate = 16000)
                input_values, labels = input_values.to(self.device), labels.to(self.device) 
                predictions = model(input_values)
                loss = self.criterion(predictions, labels)

            self.scaler.scale(loss).backward()  # Scale the loss and call backward
            self.scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned parameters are now unscaled, clips as usual.
        # You may use the same value for max_norm here as you would without gradient scaling.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            self.scaler.step(optimizer)  # Unscale gradients and call optimizer step
            self.scaler.update()  # Update the scale for next iteration
            scheduler.step(epoch + batch_idx / len(dataloader))
            
            total_loss += loss.item()
            total_acc += 1.0 - loss.item()  # Assuming accuracy is 1 - loss for this task
            
            progress_bar.set_description(f"Training Epoch {epoch+1}/{total_epochs}, Avg Loss: {total_loss/(batch_idx+1):.4f}, Acc: {total_acc/(batch_idx+1):.4f}")
            optimizer.zero_grad()
            del input_values, labels, predictions, loss
            torch.cuda.empty_cache()
        
        return total_loss / len(dataloader), total_acc / len(dataloader)

    def _evaluate(self, model, dataloader):
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_acc_flat = {
            'original': 0.0,
            'gaussian': 0.0,
            'cubic': 0.0,
            'kalman': 0.0,
        }

        with torch.no_grad():
            for input_values, labels, ground_truth_names in dataloader:
                ground_truth_labels = self._get_ground_truth_labels(ground_truth_names)
                predictions = self._process_sequences(model, input_values)
                loss = self.criterion(predictions.to("cpu"), labels)
                total_loss += loss.item()
                total_acc += 1.0 - loss.item()

                predictions_np = predictions.cpu().numpy()
                for method in ['original','gaussian',"cubic", "kalman"]:
                    average = self._unsplit_data(predictions_np, method, ground_truth_labels.shape[-1])
                    total_acc_flat[method] += self._calculate_flattened_accuracy(average, ground_truth_labels)

                del input_values, labels, predictions, loss
                torch.cuda.empty_cache()

        num_samples = len(dataloader)
        avg_loss = total_loss / num_samples
        avg_acc = total_acc / num_samples
        avg_flat_acc = {method: acc / num_samples for method, acc in total_acc_flat.items()}

        print(f"Val Loss: {avg_loss:.4f}, Val Acc: {avg_acc:.4f}")
        for method, acc in avg_flat_acc.items():
            print(f"Val Flat Acc ({method}): {acc:.4f}")

        return avg_loss, avg_acc, 0 , 0


    def _unsplit_data(self, windowed_data, method, original_length):
            original_length = original_length
            window_size = self.config.window_size
            step_size = self.config.step_size
            data_points_per_second = 25

            if method == 'original':
                return self._unsplit_data_ogsize(windowed_data, window_size, step_size, data_points_per_second, original_length)
            elif method == 'gaussian':
                return self._unsplit_data_gaussian(windowed_data, window_size, step_size, data_points_per_second, original_length)
            elif method == 'cubic':
                return self._unsplit_data_cubic(windowed_data, window_size, step_size, data_points_per_second, original_length)
            elif method == 'kalman':
                return self._unsplit_data_kalman(windowed_data, window_size, step_size, data_points_per_second, original_length)
            elif method == 'butterworth':
                return self._unsplit_data_butterworth(windowed_data, window_size, step_size, data_points_per_second, original_length)
            elif method == 'probabilistic':
                return self._unsplit_data_probabilistic(windowed_data, window_size, step_size, data_points_per_second, original_length)
            else:
                raise ValueError(f"Unknown method: {method}")

    def _get_ground_truth_labels(self, ground_truth_names):
        ground_truth_labels = []
        for batch_name in ground_truth_names:
            ground_truth_label = self._choose_real_labs_only_with_filenames(self.ground_labels, [batch_name])
            ground_truth_labels.append(ground_truth_label)
            
        return np.array(ground_truth_labels)[:, :, -1].astype(np.float32)

    def _process_sequences(self, model, input_values):
        predictions = []
        for i in range(input_values.size(1)):
    
            slice_input = input_values[:, i, :]
            slice_input = np.stack(slice_input, axis=0)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                slice_input = self.processor(slice_input, return_tensors="pt", padding="longest", sampling_rate = 16000)
                slice_input = slice_input.to(self.device)
                pred = model(slice_input)
            predictions.append(pred)
        return torch.stack(predictions, dim=1)

    def _calculate_flattened_accuracy(self, average, ground_truth_labels):
        s_acc = 0
        for b in range(len(ground_truth_labels)):
            s, _ = scipy.stats.pearsonr(average[b], ground_truth_labels[b])
            s_acc += s
        return s_acc / len(ground_truth_labels)
    
    def _choose_real_labs_only_with_filenames(self, labels, filenames):
        return labels[labels['filename'].isin(filenames)]

    def _unsplit_data_ogsize(self,windowed_data, window_size, step_size, data_points_per_second, original_length):
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


    def _unsplit_data_gaussian(self,windowed_data, window_size, step_size, data_points_per_second, original_length):

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

    def _unsplit_data_cubic(self,windowed_data, window_size, step_size, data_points_per_second, original_length):

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

    def _unsplit_data_kalman(self, windowed_data, window_size, step_size, data_points_per_second, original_length):
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
    def _unsplit_data_probabilistic(self, windowed_data, window_size, step_size, data_points_per_second, original_length):
        # Ensure windowed_data is a tensor on the GPU
        windowed_data = torch.tensor(windowed_data, device='cuda')
        batch_size, num_windows, prediction_size = windowed_data.shape
        window_size_points = int(window_size * data_points_per_second)
        step_size_points = int(step_size * data_points_per_second)

        # Initialize tensors on GPU
        original_data = torch.zeros((batch_size, original_length), device='cuda')
        overlap_count = torch.zeros((batch_size, original_length), device='cuda')
        
        def breathing_weight(x):
            return torch.exp(-x**2 / (2 * 0.3**2))  # Gaussian-like weight, adjust 0.3 as needed

        # First pass: Identify overlap regions
        for i in range(num_windows):
            start = i * step_size_points
            end = min(start + window_size_points, original_length)
            overlap_count[:, start:end] += 1

        # Second pass: Apply weighted reconstruction only in overlap regions
        for i in range(num_windows):
            start = i * step_size_points
            end = min(start + window_size_points, original_length)
            segment_length = end - start

            # Create weights for this window
            weights = breathing_weight(torch.linspace(-1, 1, segment_length, device='cuda'))
            
            # Identify overlap and non-overlap regions in this window
            is_overlap = overlap_count[:, start:end] > 1
            
            # For overlap regions: apply weighted accumulation
            original_data[:, start:end] += torch.where(
                is_overlap,
                windowed_data[:, i, :segment_length] * weights,
                torch.zeros_like(windowed_data[:, i, :segment_length])
            )
            
            # For non-overlap regions: directly copy the data
            original_data[:, start:end] = torch.where(
                ~is_overlap,
                windowed_data[:, i, :segment_length],
                original_data[:, start:end]
            )

        # Normalize only the overlapping regions
        overlap_mask = overlap_count > 1
        original_data[overlap_mask] /= overlap_count[overlap_mask]

        return original_data.cpu().numpy()
    def _unsplit_data_butterworth(self, windowed_data, window_size, step_size, data_points_per_second, original_length):
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
            original_data[:, start:end] += windowed_data[:, i, :segment_length]
            overlap_count[:, start:end] += 1

        # Average the overlapping regions
        original_data = torch.where(overlap_count != 0, original_data / overlap_count, original_data)

        # Apply Butterworth filter to overlapping regions
        for b in range(batch_size):
            # Find contiguous overlap regions
            overlap_indices = torch.nonzero(overlap_regions[b]).squeeze()
            if overlap_indices.numel() > 0:
                if overlap_indices.dim() == 0:
                    # Only one overlap point
                    starts = ends = overlap_indices.unsqueeze(0)
                else:
                    # Multiple overlap points
                    starts = overlap_indices[:-1][(overlap_indices[1:] - overlap_indices[:-1] > 1)]
                    ends = overlap_indices[1:][overlap_indices[1:] - overlap_indices[:-1] > 1]
                    
                    if len(starts) == 0:
                        # All overlap points are contiguous
                        starts = overlap_indices[0].unsqueeze(0)
                        ends = overlap_indices[-1].unsqueeze(0)
                    else:
                        # Add first and last points if they're not included
                        if overlap_indices[0] != starts[0]:
                            starts = torch.cat([overlap_indices[0].unsqueeze(0), starts])
                        if overlap_indices[-1] != ends[-1]:
                            ends = torch.cat([ends, overlap_indices[-1].unsqueeze(0)])
                
                # Apply filter to each contiguous overlap region
                for start, end in zip(starts, ends):
                    segment = original_data[b, start:end].cpu().numpy()
                    filtered_segment = sosfilt(sos, segment)
                    original_data[b, start:end] = torch.from_numpy(filtered_segment).to('cuda')

        return original_data.cpu().numpy()
        
    def sine_wave_asymmetrical(self, t, amplitude, frequency_in, frequency_out):
        """
        Generate an asymmetrical breathing pattern centered around 0.

        Parameters:
            t (np.ndarray): Time vector.
            amplitude (float): Amplitude of the wave (1 for -1 to 1).
            frequency_in (float): Frequency of inhalation (higher for faster rise).
            frequency_out (float): Frequency of exhalation (lower for slower drop).

        Returns:
            np.ndarray: Combined breathing pattern.
        """
        # Sine wave for inhalation (quick rise, from -1 to 1)
        wave_in = amplitude * (0.5 * (1 + np.sin(2 * np.pi * frequency_in * t)))  # Range 0 to 1
        
        # Linear function for exhalation, gradually decreasing (from 1 to -1)
        exhalation_duration = 1 / frequency_out  # Duration based on frequency_out
        wave_out = amplitude * (1 - (t / exhalation_duration))  # Linear drop from 1 to -1

        # Combine parts to center around 0 and oscillate from -1 to 1
        combined_wave = wave_in - (1 - wave_out)  # Inhale up, exhale down
        combined_wave = np.clip(combined_wave, -1, 1)  # Clip to the range -1 to 1

        return combined_wave

 
    def _log_metrics(self, writer, epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, fold, test_flat_acc, val_flat_acc,test_flat_acc_ola, val_flat_acc_ola):
        writer.add_scalar(f'Loss/train/fold_{fold}', train_loss, epoch)
        writer.add_scalar(f'Loss/val/fold_{fold}', val_loss, epoch)
        writer.add_scalar(f'Loss/test/fold_{fold}', test_loss, epoch)
        writer.add_scalar(f'Accuracy/train/fold_{fold}', train_acc, epoch)
        writer.add_scalar(f'Accuracy/val/fold_{fold}', val_acc, epoch)
        writer.add_scalar(f'Accuracy/test/fold_{fold}', test_acc, epoch)
        writer.add_scalar(f'Accuracy_flat/test/fold_{fold}', test_flat_acc, epoch)
        writer.add_scalar(f'Accuracy_flat/val/fold_{fold}', val_flat_acc, epoch)
        writer.add_scalar(f'Accuracy_flat_ola/test/fold_{fold}', test_flat_acc_ola, epoch)
        writer.add_scalar(f'Accuracy_flat_ola/val/fold_{fold}', val_flat_acc_ola, epoch)

    def _calculate_average_results(self, model_results):
        avg_best_val_loss = np.mean([r['best_val_loss'] for r in model_results])
        avg_test_loss = np.mean([r['test_loss'] for r in model_results])
        avg_test_acc = np.mean([r['test_acc'] for r in model_results])
        return {
            'best_val_loss': avg_best_val_loss,
            'test_loss': avg_test_loss,
            'test_acc': avg_test_acc
        }

    def _log_average_results(self, writer, avg_results):
        writer.add_scalar('Average/best_val_loss', avg_results['best_val_loss'], 0)
        writer.add_scalar('Average/test_loss', avg_results['test_loss'], 0)
        writer.add_scalar('Average/test_acc', avg_results['test_acc'], 0)

    def _print_model_results(self, model_name, model_results, avg_results):
        print(f"\nResults for {model_name}:")
        print(f"Average Best Val Loss: {avg_results['best_val_loss']:.4f}")
        print(f"Average Test Loss: {avg_results['test_loss']:.4f}")
        print(f"Average Test Accuracy: {avg_results['test_acc']:.4f}")
        for result in model_results:
            print(f"Fold {result['fold']}: Best Val Loss: {result['best_val_loss']:.4f}, Test Loss: {result['test_loss']:.4f}, Test Acc: {result['test_acc']:.4f}")
        
    def _save_model(self, model, filename):
        path = os.path.join(self.model_save_dir, filename)
        torch.save(model.state_dict(), path)
        print(f"Saved model to {path}")