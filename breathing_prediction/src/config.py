# config.py

class Config:
    # Data parameters
    audio_interspeech_norm = "/home/glenn/Downloads/ComParE2020_Breathing/wav/"
    breath_interspeech_folder = "/home/glenn/Downloads/ComParE2020_Breathing/lab/"
    window_size = 32
    step_size = 4
    n_folds = 5
    device = "cuda"

    # Model parameters
    models = {
        "VRBModel": {
            "model_name": "facebook/hubert-large-ls960",
            "hidden_units": 64,
            "n_gru": 3,
            "output_size": None  # Will be set dynamically
        },
        "Wav2Vec2ConvLSTMModel": {
            "model_name": "facebook/wav2vec2-base",
            "hidden_units": 128,
            "n_lstm": 2,
            "output_size": None  # Will be set dynamically
        },
        "RespBertLSTMModel": {
            "bert_config": "wav2vec2",
            "hidden_units": 128,
            "n_lstm": 2,
            "output": None  # Will bclass e set dynamically
        },
        "RespBertAttionModel": {
            "bert_config": "hubert",
            "hidden_units": 128,
            "n_attion": 1,
            "output": None  # Will be set dynamically
        }
    }

    # Training parameters
    epochs = 1
    batch_size = 1
    patience = 50
    learning_rate = 1e-4
    weight_decay = 1e-2

    # Optimizer and scheduler parameters
    optimizer = "AdamW"
    scheduler = "CosineAnnealingWarmRestarts"
    t0 = 10
    t_mult = 1
    min_lr = 1e-5

    # Logging and saving
    log_dir = "../results/logs"
    model_save_dir = "../results/models"

    # BERT configuration
    bert_model = "facebook/hubert-base-ls960"

    # Data processing
    frame_rate = 16000  # Based on the prepare_data function

    # Loss function
    criterion = "PearsonLoss"

    # Early stopping
    early_stopping_mode = "min"
    early_stopping_delta = 0

    # Wav2Vec2Processor
    processor_model = "facebook/wav2vec2-large-960h"

    # Other parameters
    random_seed = 42