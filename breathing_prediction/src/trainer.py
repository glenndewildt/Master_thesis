import os
import datetime
import numpy as np
import csv
import scipy
import torch
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


class Trainer:
    def __init__(self, config, model_classes, criterion, device, bert_config, ground_labels, processor):
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
        kfold = KFold(n_splits=self.config.n_folds, shuffle=True)
        
        for model_name, model_class in self.model_classes.items():
            print(f"Training {model_name}...")
            model_results = []
            
            writer = SummaryWriter(os.path.join(self.run_dir, f"{model_name}"))
            self.writer = writer
            for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
                print(f"Fold {fold + 1}/{self.config.n_folds}")

                train_sampler = SubsetRandomSampler(train_idx)
                val_sampler = SubsetRandomSampler(val_idx)
                self._log_data_used_to_csv(model_name, fold, train_idx, val_idx)
                
                train_loader = DataLoader(train_data, batch_size=self.config.batch_size, sampler=train_sampler)
                val_loader = DataLoader(train_data, batch_size=self.config.batch_size, sampler=val_sampler)
                test_loader = DataLoader(test_data, batch_size=1)

                model_config = self.config.models[model_name]
                model_config['output_size'] = train_data.get_output_shape()

                model = model_class(bert_config=self.bert_config, config=model_config).to(self.device)
                #Load from path
                model.load_state_dict(torch.load("../results/logs/run_20240926_152058/RespBertAttionModel_best_model_fold_0.pt", map_location=self.device))

                optimizer = AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.config.t0, T_mult=self.config.t_mult, eta_min=self.config.min_lr)
                
                best_val_loss = float('inf')
                best_model_path = None
                early_stopping = EarlyStopping(patience=self.config.patience, mode='min')
                for epoch in range(self.config.epochs):

                    train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, scheduler, epoch, self.config.epochs)
                    
                    val_loss, val_acc , val_flat_acc = self._evaluate(model, val_loader)
                    test_loss, test_acc, test_flat_acc = self._evaluate(model, test_loader)

                    
                    self._log_metrics(writer, epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, fold,test_flat_acc, val_flat_acc)
                    
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

    def _train_epoch(self, model, dataloader, optimizer, scheduler, epoch, total_epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        
        input_values, labels = flatten_and_shuffle_data(dataloader)
        train_dataset = AugmentedDataset(input_values, labels)
        dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=8)   
        l = len(dataloader.dataset)
            
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{total_epochs}")
        i = 0

        for batch_idx, (input_values, labels) in enumerate(progress_bar):

            optimizer.zero_grad()
            input_values = self.processor(input_values, return_tensors="pt", padding="longest", sampling_rate = 16000).input_values
            input_values = input_values.reshape(input_values.shape[1], input_values.shape[-1])
            input_values, labels = input_values.to(self.device), labels.to(self.device)
          
            predictions = model(input_values)
            loss = self.criterion(predictions.float(), labels.float())
            if i == 0:
                #make_dot(predictions, params=dict(model.named_parameters())).render("model", format="png")
                i =1  
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + batch_idx / len(dataloader))
            
            total_loss += loss.item()
            total_acc += 1.0 - loss.item()  # Assuming accuracy is 1 - loss for this task
            
            progress_bar.set_description(f"Training Epoch {epoch+1}/{total_epochs}, Avg Loss: {total_loss/(batch_idx+1):.4f}, Acc: {total_acc/(batch_idx+1):.4f}")
            
            del input_values, labels, predictions, loss
            torch.cuda.empty_cache()
        
        return total_loss / len(dataloader), total_acc / len(dataloader)

    def _evaluate(self, model, dataloader):
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_acc_flat = 0.0
        
        with torch.no_grad():
            for input_values, labels, ground_truth_names in dataloader:
                input_values = self.processor(input_values, return_tensors="pt", padding="longest", sampling_rate = 16000).input_values
                input_values = input_values.reshape(input_values.shape[1],input_values.shape[2], input_values.shape[-1])
                input_values, labels = input_values.to(self.device), labels.to(self.device)
                                
                ground_truth_labels = self._get_ground_truth_labels(ground_truth_names)
                
                predictions = self._process_sequences(model, input_values)
                loss = self.criterion(predictions, labels)
                
                total_loss += loss.item()
                total_acc += 1.0 - loss.item() 
                
                average = self._unsplit_data_ogsize(predictions.cpu().numpy(), self.config.window_size, self.config.step_size, self.config.data_points_per_second, ground_truth_labels.shape[-1])
                total_acc_flat += self._calculate_flattened_accuracy(average, ground_truth_labels)
                
                del input_values, labels, predictions, loss
                torch.cuda.empty_cache()
        
        num_samples = len(dataloader)

        avg_loss, avg_acc, avg_flat_acc = total_loss / num_samples, total_acc / num_samples, total_acc_flat / num_samples
        print(f"val loss {avg_loss}, val_acc {avg_acc} , val_flat_acc, {avg_flat_acc}" )
        return avg_loss, avg_acc, avg_flat_acc 


    def _get_ground_truth_labels(self, ground_truth_names):
        ground_truth_labels = []
        for batch_name in ground_truth_names:
            ground_truth_label = self._choose_real_labs_only_with_filenames(self.ground_labels, [batch_name])
            ground_truth_labels.append(ground_truth_label)
            
        return np.array(ground_truth_labels)[:, :, -1].astype(np.float32)

    def _process_sequences(self, model, input_values):
        predictions = []
        for i in range(input_values.size(1)):
            input_slice = input_values[:, i, :]
            pred = model(input_slice.float())
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

    def _unsplit_data_ogsize(self, windowed_data, window_size, step_size, data_points_per_second, original_length):
        batch_size, num_windows, prediction_size = windowed_data.shape
        window_size_points = window_size * data_points_per_second
        step_size_points = step_size * data_points_per_second
        original_data = np.zeros((batch_size, original_length))
        overlap_count = np.zeros((batch_size, original_length))
        
        for b in range(0,batch_size ):
            for i in range(0,num_windows):
                start = i * step_size_points
                end = start + window_size_points
                if end > original_length:
                    end = original_length
                segment_length = end - start
                original_data[b, start:end] += windowed_data[b, i, :segment_length]
                overlap_count[b, start:end] += 1
        
        # Average the overlapping regions
        original_data = np.divide(original_data, overlap_count, where=overlap_count != 0)
        
        # Trim the data to match the original length
        original_data = original_data[:, :original_length]
        
        return original_data
    
    def _log_metrics(self, writer, epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, fold, test_flat_acc, val_flat_acc):
        writer.add_scalar(f'Loss/train/fold_{fold}', train_loss, epoch)
        writer.add_scalar(f'Loss/val/fold_{fold}', val_loss, epoch)
        writer.add_scalar(f'Loss/test/fold_{fold}', test_loss, epoch)
        writer.add_scalar(f'Accuracy/train/fold_{fold}', train_acc, epoch)
        writer.add_scalar(f'Accuracy/val/fold_{fold}', val_acc, epoch)
        writer.add_scalar(f'Accuracy/test/fold_{fold}', test_acc, epoch)
        writer.add_scalar(f'Accuracy_flat/test/fold_{fold}', test_flat_acc, epoch)
        writer.add_scalar(f'Accuracy_flat/val/fold_{fold}', val_flat_acc, epoch)

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