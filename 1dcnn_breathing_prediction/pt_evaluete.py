import os
import pandas as pd
import numpy as np
from pt_utils import *
from pt_dataset import *
from pt_models import *
from pt_utils import *
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from transformers import Wav2Vec2Processor,Wav2Vec2FeatureExtractor,AutoModel
from tensorboardX import SummaryWriter
from pt_utils import load_data, prepare_data, reshaping_data_for_model, unsplit_data_ogsize
from pt_dataset import BreathingDataset
import scipy.stats
from torch.cuda.amp import autocast
import glob
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

def create_run_directory():
    base_dir = "pt_eval_batch"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def evaluate_model(model_path, test_loader, config, device, prepared_test_labels_timesteps, test_labels, test_dict, window_size, step_size):
    # Load model
    model = config["model"](config)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    test_pred = []
    test_loss = 0.0
    progress_bar = tqdm(test_loader, desc=f"Testing {os.path.basename(model_path)}")

    with torch.no_grad():
        for batch_d, batch_lbs in progress_bar:
            with torch.amp.autocast(device_type="cuda"):
                batch_d = batch_d.to(device)
                batch_lbs = batch_lbs.to(device)
                batch_d = model(batch_d)
                loss = correlation_coefficient_loss(batch_d, batch_lbs)
            
            test_loss += loss.item()
            test_pred.extend(batch_d.float().cpu().numpy())
            progress_bar.set_postfix({'test loss': f'{test_loss/(progress_bar.n+1):.4f}'})
            
            del loss, batch_d, batch_lbs
            torch.cuda.empty_cache()

    test_loss /= len(test_loader)
    test_pred = np.array(test_pred).reshape(prepared_test_labels_timesteps.shape)
    test_ground_truth = _get_ground_truth_labels(list(test_dict.values()), test_labels)
    
    # Calculate accuracies for different methods
    accuracies = {}
    for method in ['original', 'gaussian']:
        average = unsplit_data(test_pred, window_size, step_size, method, test_ground_truth.shape[-1])
        accuracy = _calculate_flattened_accuracy(average, test_ground_truth)
        accuracies[method] = accuracy
    
    return test_loss, accuracies, average, test_ground_truth

def batch_evaluate(path_to_test_data, path_to_test_labels, models_folder, window_size=30, step_size=6, batch_size=10, config=None, processor= None):
    run_dir = create_run_directory()
    results_path = os.path.join(run_dir, 'evaluation_results.csv')
    
    # Load and prepare test data
    test_data, test_labels, test_dict, frame_rate = load_data(path_to_test_data, path_to_test_labels, 'test')
    prepared_test_data, prepared_test_labels, prepared_test_labels_timesteps = prepare_data(
        test_data, test_labels, test_dict, frame_rate, size_window=window_size * 16000, step_for_window=step_size * 16000
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["output_size"] = prepared_test_labels.shape[-1]
    
    # Reshape data
    test_d, test_lbs = reshaping_data_for_model(prepared_test_data, prepared_test_labels)
    print(f"Test data shape: {test_d.shape}")
    
    # Create dataset and DataLoader
    test_dataset = BreathingDataset(test_d, test_lbs, processor, window_size, step_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn)
    
    # Find all model files
    model_files = glob.glob(os.path.join(models_folder, 'best_model_fold*'))
    results = []
    total_pred = None
    count = 0
    test_ground_truth = None
    # Evaluate each model
    for model_path in model_files:
        count +=1
        fold_num = int(model_path.split('fold')[-1])
        test_loss, accuracies, pred, test_ground_truth = evaluate_model(
            model_path, test_loader, config, device,
            prepared_test_labels_timesteps, test_labels,
            test_dict, window_size, step_size
        )
        test_ground_truth = test_ground_truth
        if total_pred is None:
            total_pred = pred
        else:
            total_pred += pred
        
        # Store results
        result = {
            'fold': fold_num,
            'model_path': model_path,
            'test_loss': test_loss
        }
        result.update({f'accuracy_{method}': acc for method, acc in accuracies.items()})
        results.append(result)
        
        print(f"\nFold {fold_num} Results:")
        print(f"Test Loss: {test_loss:.4f}")
        for method, acc in accuracies.items():
            print(f"Accuracy ({method}): {acc:.4f}")
    total_pred = total_pred / count          
    accuracies = {}
    for method in ['original', 'gaussian']:
        accuracy = _calculate_flattened_accuracy(total_pred, test_ground_truth)
        accuracies[method] = accuracy
    for method, acc in accuracies.items():
        print(f"Accuracy ({method}): {acc:.4f}")
        

    
    # Create DataFrame and calculate averages
    results_df = pd.DataFrame(results)
    averages = results_df.select_dtypes(include=[np.number]).mean()
    std_devs = results_df.select_dtypes(include=[np.number]).std()
    
    # Add average and std dev rows
    results_df.loc['average'] = averages
    results_df.loc['std_dev'] = std_devs
    
    # Save results
    results_df.to_csv(results_path, index=True)
    print(f"\nResults saved to {results_path}")
    
    return results_df

if __name__ == "__main__":

    # Model configuration (using RespBertCNNModel as example)
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
        },
             "RespBertCNNModel_skip": {
            'model' : RespBertCNNModel_skip,
            "model_name": "microsoft/wavlm-large",
            "hidden_units": 256,
            "output_size": None  
        },
            "WavLMCNNLSTM": {
            'model' : WavLMCNNLSTM,
            "model_name": "microsoft/wavlm-large",
            "hidden_units": 128,
            "n_lstm": 2,
            "output_size": None  # Will be set dynamically
        },
            
            "WALMLSTM": {
            'model': WALMLSTM,
            "model_name": "facebook/hubert-large-ls960-ft",
            "hidden_units": 128,
            "n_lstm": 2,
            "output_size": None  
        },
            "WavlmCNNModel": {
            'model' : WavlmCNNModel,
            "model_name": "microsoft/wavlm-large",
            "hidden_units": 256,
            "output_size": None 
            },
            "RespBertCNN_12_Model": {
            'model' : RespBertCNN_12_Model,
            "model_name": "microsoft/wavlm-large",
            "hidden_units": 256,
            "output_size": None 
            },
            "HuBertCNN_12_Model": {
            'model' : HuBertCNN_12_Model,
            "model_name": "facebook/hubert-large-ls960-ft",
            "hidden_units": 256,
            "output_size": None 
            },
            "HuBertCNN_12_Model": {
            'model' : HuBertCNN_12_Model,
            "model_name": "facebook/hubert-large-ls960-ft",
            "hidden_units": 512,
            "output_size": None 
            },
            "HuBertCNN_12_Model": {
            'model' : HuBertCNN_12_Model,
            "model_name": "facebook/hubert-large-ll60k",
            "hidden_units": 256,
            "output_size": None 
            },
                        "wavCNN_12_Model": {
            'model' : HuBertCNN_12_Model,
            "model_name": "microsoft/wavlm-large",
            "hidden_units": 256,
            "output_size": None 
            },
    }
    
    path = "../DATA/"
    path = "../DATA/ComParE2020_Breathing/"

    models_folder = './pt_runs/WavLm_CnNN_16/'  # Folder containing all model files
    path_to_data=path+"/wav/"
    path_to_labels=path+"/lab/"
    
    # Evaluation parameters
    window_size = 30
    step_size = 6
    batch_size = 50
    config = model_config["wavCNN_12_Model"]
    #processor = AutoProcessor.from_pretrained(config["model_name"])
    processor = Wav2Vec2FeatureExtractor.from_pretrained(config["model_name"])

    results = batch_evaluate(
        path_to_test_data=path_to_data,
        path_to_test_labels=path_to_labels,
        models_folder=models_folder,
        window_size=window_size,
        step_size=step_size,
        batch_size=batch_size,
        config=config,
        processor=processor
    )

