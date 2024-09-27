# Local imports
from models import *
from utils import *
from losses import *
from dataset import *
from trainer import *
from config import Config

if __name__ == "__main__":
    config = Config()
    #bert_config = HubertConfig.from_pretrained(config.bert_model)
    bert_config = WavLMConfig.from_pretrained("microsoft/wavlm-large")
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-large-960h")
    #processor = WavLMProcessor.from_pretrained("microsoft/wavlm-base-plus")
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu")
    criterion = PearsonLoss()
    print(device)

    # Enable Flash Attention

    model_classes = {
        # "VRBModel": VRBModel,
        # "wav2vec2_1DCNN" : Model1DCNN,
        # "OG_1DCNN" : Model1DCNN,
        # "RespBertLSTMModel": RespBertLSTMModel,time
        # "Wav2Vec2ConvLSTMModel": Wav2Vec2ConvLSTMModel,
        "RespBertAttionModel": RespBertAttionModel,
    }

    # Prepare data
    train_data, test_data, ground_labels = prepare_data_model(
        config.audio_interspeech_norm,
        config.breath_interspeech_folder,
        window_size=config.window_size,
        step_size=config.step_size,
    )

    trainer = Trainer(config, model_classes, criterion,
                      device, bert_config, ground_labels, processor)
    trainer.train(train_data, test_data)
