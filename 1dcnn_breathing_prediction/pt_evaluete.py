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

def evaluate(path_to_test_data, path_to_test_labels, window_size=16, step_size=6,model_path= None, batch_size=10, config=None, model = None, processor=None):
    run_dir = create_run_directory()
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Parameters
    length_sequence = window_size 
    step_sequence = step_size

    # Load and prepare test data
    test_data, test_labels, test_dict, frame_rate = load_data(path_to_test_data, path_to_test_labels, 'test')
    prepared_test_data, prepared_test_labels, prepared_test_labels_timesteps = prepare_data(test_data, test_labels, test_dict, frame_rate, length_sequence * 16000, step_sequence * 16000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config["output_size"] = prepared_test_labels.shape[-1]
    writer = SummaryWriter(log_dir=os.path.join(log_dir, config["model_name"]))

    # Reshape data
    test_d, test_lbs = reshaping_data_for_model(prepared_test_data, prepared_test_labels)
    
    print(f"Test data shape: {test_d.shape}")

    # Create dataset and DataLoader
    test_dataset = BreathingDataset(test_d, test_lbs, processor, window_size, step_sequence)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=3, collate_fn=test_dataset.collate_fn)
    # Load model
    model = config["model"](config)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model = model.half()  # Convert model to float16

    # Evaluate model on test data
    model.eval()
    test_pred = []
    test_loss = 0.0
    progress_bar = tqdm(test_loader, desc=f"Test")

    with torch.no_grad():
        for batch_d, batch_lbs in progress_bar:

            
            # Use automatic mixed precision
            with torch.amp.autocast(device_type="cuda"):
                # Convert input to float16
                batch_d = batch_d.to(device)
                batch_lbs = batch_lbs.to(device)
                batch_d = model(batch_d)
                loss = correlation_coefficient_loss(batch_d, batch_lbs)
            
            test_loss += loss.item()
            # Convert back to float32 for numpy
            test_pred.extend(batch_d.float().cpu().numpy())
            
            progress_bar.set_postfix({'test loss: ': f'{test_loss/(progress_bar.n+1):.4f}'})
            
            del loss, batch_d, batch_lbs
            torch.cuda.empty_cache()


        test_loss /= len(test_loader)
        test_pred = np.array(test_pred).reshape(prepared_test_labels_timesteps.shape)
        test_ground_truth = _get_ground_truth_labels(list(test_dict.values()), test_labels)
        total_acc_flat = {
            'original': 0.0,
            'gaussian': 0.0,
            'cubic': 0.0,
            'kalman': 0.0,
        }
        predictions_np = test_pred
        for method in ['original','gaussian',"cubic", "kalman"]:
            average = unsplit_data(predictions_np,window_size, step_size, method, test_ground_truth.shape[-1])
            total_acc_flat[method] += _calculate_flattened_accuracy(average, test_ground_truth)
            print(total_acc_flat[method])



        print("\nEvaluation completed.")
        print(f"Test Loss: {test_loss:.4f}")
        #print(f"Test Pearson Coefficient (flattened): {total_acc_flat["original"]}")

        # Log test metrics
        writer.add_scalar("Test/loss", test_loss, 0)
        writer.add_scalar("Test/pearson_coef", total_acc_flat["original"], 0)

        # Log the test metrics as a table
        test_table = "| Metric | Value |\n" \
                    "|--------|-------|\n" \
                    f"| Test Loss | {test_loss:.4f} |\n" \
                    #f"| Test Pearson Coefficient | {total_acc_flat["original"]:.4f} |\n"
        writer.add_text("Test_Metrics", test_table)

        writer.close()

        # Save results to CSV
        results_df = pd.DataFrame({
            'Test_Loss': [test_loss],
            'Test_Pearson_Coefficient': [total_acc_flat["original"]]
        })
        csv_path = os.path.join(run_dir, 'test_results.csv')
        results_df.to_csv(csv_path, index=False)

        print(f"Results saved to {csv_path}")
        


if __name__ == "__main__":
    path = "/home/glenn/Downloads/"
    #path = "../DATA/"


    # Model parameters
    model_config = {
        "RespBertCNNModel": {
            'model' : RespBertCNNModel,
            "model_name": "microsoft/wavlm-large",
            "hidden_units": 256,
            "output_size": None  
        }
    }

    # Evaluation parameters
    window_size = 30
    step_size = 25
    batch_size = 4
    
    config = model_config["RespBertCNNModel"]
    #processor = Wav2vec2F.from_pretrained(config["model_name"])
    processor = Wav2Vec2FeatureExtractor.from_pretrained(config["model_name"])

    # Create and initialize model
    model_folder = "/home/glenn/Downloads/pt_runs/pt_runs/"

    # Load the pre-trained model weights
    model_path = model_folder+"Wavml_cnn_full/best_model"  # Update this path

    evaluate(
        path_to_test_data=path+"ComParE2020_Breathing/wav/",
        path_to_test_labels=path+"ComParE2020_Breathing/lab/",
        window_size=window_size,
        step_size=step_size,
        batch_size=batch_size,
        config=config,
        model=None,model_path= model_path,
        processor=processor
    )