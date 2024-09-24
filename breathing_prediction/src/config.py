# config.py

class Config:
    # Data parameters
    home = "/home/glenn/Downloads"
    #home = "/home/gdwildt/Master_thesis/DATA"

    audio_interspeech_norm = home+"/ComParE2020_Breathing/wav/"
    breath_interspeech_folder = home+"/ComParE2020_Breathing/lab/"
    window_size = 16
    step_size = 4
    n_folds = 5
    device = "cuda"
    data_points_per_second = 25

    # Model parameters
    models = {
        "VRBModel": {
            "model_name": "facebook/hubert-large-ls960-ft",
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
            "hidden_units": 128,
            "n_lstm": 2,
            "output_size": None  
        },
        "RespBertAttionModel": {
            "hidden_units": 128,
            "n_attion": 1,
            "output_size": None  
        },
        "OG_1DCNN":{
            "in_channels": 1,
            "conv_layers": [
                {"out_channels": 64, "kernel_size": 10, "stride": 1, "pool_size": 10},
                {"out_channels": 128, "kernel_size": 8, "stride": 1, "pool_size": 4},
                {"out_channels": 256, "kernel_size": 6, "stride": 1, "pool_size": 4},
                {"out_channels": 256, "kernel_size": 5, "stride": 1, "pool_size": 4}
            ],
            "dropout_rate": 0.3,
            "lstm_input_size": 256,  # This should match the output size of the last Conv1D layer
            "lstm_hidden_size": 256,
            "output_size": 1,
            "n_lstm": 2

        },
        
        "wav2vec2_1DCNN": {
        "in_channels": 1,
        "conv_layers": [
            {"out_channels": 512, "kernel_size": 10, "stride": 5},
            {"out_channels": 512, "kernel_size": 3, "stride": 2},
            {"out_channels": 512, "kernel_size": 3, "stride": 2},
            {"out_channels": 512, "kernel_size": 3, "stride": 2},
            {"out_channels": 512, "kernel_size": 3, "stride": 2},
            {"out_channels": 512, "kernel_size": 2, "stride": 2},
            {"out_channels": 512, "kernel_size": 2, "stride": 2}
        ],
        "dropout_rate": 0.3,
        "lstm_hidden_size": 512,
        "dense_units": 512,
        "output_size": 1,
        "n_lstm": 2

    }
    }


    epochs = 20
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