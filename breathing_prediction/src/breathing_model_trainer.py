# Local imports
from models import *
from utils import *
from losses import *
from dataset import *
from trainer import *
from config import Config
import os



if __name__ == "__main__":

    config = Config()


    model_classes = {
        ## Recreated models
        #"VRBModel": VRBModel, #harma paper
        "Wav2Vec2ConvLSTMModel": Wav2Vec2ConvLSTMModel, #apple paper
        
        ##own proposed models useing wavml large
        #"RespBertLSTMModel": RespBertLSTMModel,
        #"RespBertAttionModel": RespBertAttionModel,
        ## test with print statements so you can see what happends
        #"RespBertLSTMModel": RespBertLSTMModelTEST,

    }

    # Prepare data
    train_data, test_data, ground_labels = prepare_data_model(
        config.audio_interspeech_norm,
        config.breath_interspeech_folder,
        window_size=config.window_size,
        step_size=config.step_size,
    )

    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu")
    criterion = PearsonLoss()
    print(device)

    trainer = Trainer(config, model_classes, criterion,
                      device, ground_labels)
    trainer.train(train_data, test_data)
