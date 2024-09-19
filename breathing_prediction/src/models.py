import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Model, 
    Wav2Vec2PreTrainedModel, 
    HubertModel, 
    HubertPreTrainedModel,
    WavLMModel,
    Wav2Vec2Processor
)
import math
import json
  



class Wav2Vec2ConvLSTMModel(nn.Module):
    def __init__(self, bert_config = None, config = None):
        super(Wav2Vec2ConvLSTMModel, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config['model_name'])
        
        for param in self.wav2vec2.parameters():
            param.requires_grad = False    
        self.input_features = self.wav2vec2.config.hidden_size       
        self.conv = nn.Conv1d(in_channels=self.input_features,
                              out_channels=self.input_features,
                              kernel_size=3,
                              padding=1)       
        self.relu = nn.ReLU() 
        self.lstm = nn.LSTM(input_size=self.input_features,
                            hidden_size=config['hidden_units'],
                            num_layers=config['n_lstm'],
                            batch_first=True)
        self.embedding = nn.Linear(config['hidden_units'], config['hidden_units'])
        self.output = nn.Linear(config['hidden_units'], config['output_size'])

        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()

    def forward(self, input_values):

        wav2vec2_outputs = self.wav2vec2(input_values)
        
        features = wav2vec2_outputs[0]
        x = features.permute(0, 2, 1)
        
        x = self.conv(x)
        
        x = self.relu(x)
        
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        
        last_time_step = lstm_out[:, -1, :]
        
        embed = self.embedding(last_time_step)
        
        output = self.output(embed)
        
        x = self.tanh(output)
        
        x = self.flatten(x)
        
        return x
    
class VRBModel(nn.Module):
    def __init__(self, bert_config = None, config = None):
        super(VRBModel, self).__init__()

        self.hubert = HubertModel.from_pretrained(config['model_name'])
        for param in self.hubert.parameters():
            param.requires_grad = False    
        self.input_features = self.hubert.config.hidden_size       

        self.gru = nn.GRU(input_size=self.input_features,
                          hidden_size=config['hidden_units'],
                          num_layers=config['n_gru'],
                          batch_first=True)
        self.embedding = nn.Linear(config['hidden_units'], config['output_size'])
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()       
    
    def forward(self, input_values, attention_mask=None):
        hubert_outputs = self.hubert(input_values, attention_mask=attention_mask)
        features = hubert_outputs[0]        
        gru_out, _ = self.gru(features)      
        last_time_step = gru_out[:, -1, :]     
        embed = self.embedding(last_time_step)
        x = self.tanh(embed)
        x = self.flatten(x)
        return x

class RespBertLSTMModel(Wav2Vec2PreTrainedModel):
    def __init__(self, bert_config,config):
        super().__init__(bert_config)
        self.config = bert_config
        self.output = config['output']
        
        if config['model_type'] == "wav2vec2":
            self.wav_model = Wav2Vec2Model(bert_config)
        elif config['model_type'] == "hubert":
            self.wav_model = HubertModel(bert_config)
        elif config['model_type'] == "wavlm":
            self.wav_model = WavLMModel(bert_config)
        else:
            raise ValueError("Unsupported model type")

        self.d_model = bert_config.hidden_size
        self.features = config['hidden_units']
        
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.features, num_layers=config['n_lstm'], batch_first=True)
        
        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.features, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(),
            nn.Dropout(0.2),  

            nn.Conv1d(self.features, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(),
            nn.Dropout(0.2),  
        )

        self.feature_downsample = nn.Linear(self.features, 1)
        self.time = NotImplementedError()
        self.tanh_va = nn.Tanh()
        self.flatten = nn.Flatten()
        
        self.unfreeze_last_n_blocks(4)
                
    def freeze_conv_only(self):
        for param in self.wav_model.feature_extractor.conv_layers.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, num_blocks: int) -> None:
        for param in self.wav_model.parameters():
            param.requires_grad = False

        for i in range(0, num_blocks):
            for param in self.wav_model.encoder.layers[-1 * (i + 1)].parameters():
                param.requires_grad = True

    def forward(self, input_values):
        input_values = input_values.float()       
        x = self.wav_model(input_values)[0]               
        x, _ = self.lstm(x)       
        x = x.permute(0, 2, 1)        
        x = self.time_downsample(x)  
        x = x.permute(0, 2, 1)
        x = self.feature_downsample(x)
        x = self.flatten(x)
        self.time = nn.Linear(x.shape[-1], self.output)
        x = self.time(x)
        x = self.tanh_va(x)
        return x

class RespBertAttionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, bert_config, config):
        super().__init__(bert_config)
        self.config = bert_config
        self.output = config['output']
        
        if config['model_type'] == "wav2vec2":
            self.wav_model = Wav2Vec2Model(bert_config)
        elif config['model_type'] == "hubert":
            self.wav_model = HubertModel(bert_config)
        elif config['model_type'] == "wavlm":
            self.wav_model = WavLMModel(bert_config)
        else:
            raise ValueError("Unsupported model type")

        self.d_model = bert_config.hidden_size
        self.features = config['hidden_units']
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=12, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=config['n_attion'])

        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.d_model, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(),
            nn.Dropout(0.2),  

            nn.Conv1d(self.features, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(),
            nn.Dropout(0.2),  
        )

        self.feature_downsample = nn.Linear(self.features, 1)
        self.time = NotImplementedError()
        self.tanh_va = nn.Tanh()
        self.flatten = nn.Flatten()      
        self.init_weights()
        self.unfreeze_last_n_blocks(4)
                
    def freeze_conv_only(self):
        for param in self.wav_model.feature_extractor.conv_layers.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, num_blocks: int) -> None:
        for param in self.wav_model.parameters():
            param.requires_grad = False

        for i in range(0, num_blocks):
            for param in self.wav_model.encoder.layers[-1 * (i + 1)].parameters():
                param.requires_grad = True

    def forward(self, input_values):
        input_values = input_values.float()      
        x = self.wav_model(input_values)[0]        
        x = self.transformer_layer(x)        
        x = x.permute(0, 2, 1)       
        x = self.time_downsample(x)  
        x = x.permute(0, 2, 1)     
        x = self.feature_downsample(x)
        x = self.flatten(x)
        self.time = nn.Linear(x.shape[-1], self.output)
        x = self.time(x)
        x = self.tanh_va(x)
        return x
    
    
