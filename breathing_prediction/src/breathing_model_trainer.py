# Local imports
from models import *
from utils import *
from losses import *
from dataset import *
from trainer import *
from config import Config

if __name__ == "__main__":
    config = Config()
    # Define the model name

    # Load the processor

    #bert_config = HubertConfig.from_pretrained(config.bert_model)

    # Enable Flash Attention

    model_classes = {
        # "VRBModel": VRBModel,
        # "wav2vec2_1DCNN" : Model1DCNN,
        # "OG_1DCNN" : Model1DCNN,
        "RespBertLSTMModel": RespBertLSTMModel,
        # "Wav2Vec2ConvLSTMModel": Wav2Vec2ConvLSTMModel,
        #"RespBertAttionModel": RespBertAttionModel,
    }

    # Prepare data
    train_data, test_data, ground_labels = prepare_data_model(
        config.audio_interspeech_norm,
        config.breath_interspeech_folder,
        window_size=config.window_size,
        step_size=config.step_size,
    )

    model_name = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
    #model_name = "microsoft/wavlm-large"
    #bert_config = AutoConfig.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    bert_config = AutoConfig.from_pretrained(model_name )
    #processor = AutoProcessor.from_pretrained(model_name)

    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu")
    criterion = PearsonLoss()
    print(device)

    trainer = Trainer(config, model_classes, criterion,
                      device, bert_config, ground_labels, processor)
    trainer.train(train_data, test_data)
