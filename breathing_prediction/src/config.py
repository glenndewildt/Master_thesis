from transformers import (
    AutoModel
)
class Config:

    # Data parameters
    home = "/home/glenn/Downloads"
    #home = "../../DATA"

    audio_interspeech_norm = home+"/ComParE2020_Breathing/wav/"
    breath_interspeech_folder = home+"/ComParE2020_Breathing/lab/"
    window_size = 30
    step_size = 4
    n_folds = 7
    device = "cuda"
    data_points_per_second = 25
    
    ## Train parameters
    epochs = 60
    batch_size = 5
    patience = 15
    learning_rate = 5e-4
    weight_decay = 0.01

    # Optimizer and scheduler parameters
    optimizer = "AdamW"
    scheduler = "CosineAnnealingWarmRestarts"
    t0 = 10
    t_mult = 2
    min_lr = 2e-5

    # Logging and saving
    log_dir = "../results/logs"
    model_save_dir = "../results/models"

    # Data processing
    frame_rate = 16000  # Based on the prepare_data function

    # Loss function
    criterion = "PearsonLoss"

    # Early stopping
    early_stopping_mode = "min"
    early_stopping_delta = 0

    # Other parameters
    random_seed = 42
    
    ## ##Bert based models used for getting the features of a model
    ##Apple procesor with 7 layers
    apple_encoder = AutoModel.from_pretrained("facebook/wav2vec2-base")
    apple_encoder.encoder.layers = apple_encoder.encoder.layers[0:6]
    
    harma_encoder = AutoModel.from_pretrained("facebook/hubert-large-ls960-ft")

    # Model parameters
    models = {
        "VRBModel": {
            "model_name": "facebook/hubert-large-ls960-ft",
            "encoder": harma_encoder,
            "hidden_units": 64,
            "n_gru": 3,
            "output_size": None  # Will be set dynamically
        },
        "Wav2Vec2ConvLSTMModel": {
            "encoder": apple_encoder,
            "model_name": "facebook/wav2vec2-base",
            "hidden_units": 128,
            "n_lstm": 2,
            "output_size": None  # Will be set dynamically
        },
        "RespBertLSTMModel": {
            "encoder": None,
            "model_name": "microsoft/wavlm-large",
            "hidden_units": 128,
            "n_lstm": 2,
            "output_size": None  
        },
        "RespBertAttionModel": {
            "encoder": None,
            "model_name": "microsoft/wavlm-large",
            "hidden_units": 128,
            "n_attion": 1,
            "output_size": None  
        }
    }


