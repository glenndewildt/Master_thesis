import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Model, 
    Wav2Vec2PreTrainedModel, 
    HubertModel, 
    HubertPreTrainedModel,
    WavLMModel,
    Wav2Vec2Processor,
    HubertConfig,
    WavLMForCTC, 
)
import math
import json




##BASED IN APPLE PAPER
class Wav2Vec2ConvLSTMModel(nn.Module):
    def __init__(self, bert_config = None, config = None):
        super(Wav2Vec2ConvLSTMModel, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config['model_name'])
        self.wav2vec2.encoder.layers = self.wav2vec2.encoder.layers[0:6] 
        self.freeze_blocks()
        
        self.input_features = self.wav2vec2.config.hidden_size       
        self.conv = nn.Conv1d(in_channels=self.input_features,
                              out_channels=self.input_features,
                              kernel_size=3,
                              padding=1, dilation = 1)
               
        self.lstm = nn.LSTM(input_size=self.input_features,
                            hidden_size=config['hidden_units'],
                            num_layers=config['n_lstm'],
                            batch_first=True)
        self.embedding = nn.Linear(config['hidden_units'], config['hidden_units'])
        self.output = nn.Linear(config['hidden_units'], config['output_size'])

        self.flatten = nn.Flatten()
    def freeze_blocks(self, num_blocks: int):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
    

    def forward(self, input_values):
        wav2vec2_outputs = self.wav2vec2(input_values)     
        features = wav2vec2_outputs[0] # [0] gets the transformer features in this case from the 7 the layer
        x = features.permute(0, 2, 1)         
        x = self.conv(x) # goes finds patterns in the features over all for breathing features for each timestep      
        x = x.permute(0, 2, 1)      
        lstm_out, _ = self.lstm(x) # for each time step there are now 128 features into a lstm       
        last_time_step = lstm_out[:, -1, :]  # get the lest timestep to get the the timestep with all the incorparated data from the other steps   
        embed = self.embedding(last_time_step)   # a linear layer with a dimention of of 128   
        output = self.output(embed)    # last layer goes from 128 from the embedding layer to 400 in the case of 30 second window          
        x = self.flatten(output)
        
        return x
    
##BASED ON VRB HARMA2023 PAPER    
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
    
## MY PROPOSED MODEL DESIGNS
class RespBertLSTMModel(Wav2Vec2PreTrainedModel):
    def __init__(self, bert_config,config = None):
        super().__init__(bert_config)
        self.config = bert_config

        self.output = config['output_size']
        
        if bert_config.model_type == "wav2vec2":
            self.wav_model = Wav2Vec2Model(bert_config)
        elif bert_config.model_type == "hubert":
            self.wav_model = HubertModel(bert_config)
        elif bert_config.model_type == "wavlm":
            self.wav_model = WavLMModel(bert_config)
        else:
            raise ValueError("Unsupported model type")
        self.wav_model.encoder.layers = self.wav_model.encoder.layers[0:8] 

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
        self.time = None
        self.tanh_va = nn.Tanh()
        self.flatten = nn.Flatten()
        
        self.unfreeze_last_n_blocks(2)
                
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
        x = self.wav_model(input_values)[0]               
        x, _ = self.lstm(x)       
        x = x.permute(0, 2, 1)        
        x = self.time_downsample(x)  
        x = x.permute(0, 2, 1)
        x = self.feature_downsample(x)
        x = self.flatten(x)
        if self.time == None:
            self.time = nn.Linear(x.shape[-1], self.output).to("cuda")
        x = self.time(x)
        x = self.tanh_va(x)
        return x

class RespBertAttionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, bert_config, config):
        super().__init__(bert_config)
        self.config = bert_config
        self.output = config['output_size']
        
        if bert_config.model_type == "wav2vec2":
            self.wav_model = Wav2Vec2Model(bert_config)
        elif bert_config.model_type == "hubert":
            self.wav_model = HubertModel(bert_config)
        elif bert_config.model_type == "wavlm":
            self.wav_model = WavLMModel(bert_config)
        else:
            raise ValueError("Unsupported model type")
        # Replace the attention mechanism within each encoder layer
        self.wav_model.encoder.layers = self.wav_model.encoder.layers[6]

        self.d_model = bert_config.hidden_size
        self.features = config['hidden_units']
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=16, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=config['n_attion'])

        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.d_model, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Conv1d(self.features, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(),            
            nn.AdaptiveAvgPool1d(self.output)  
        )

        self.feature_downsample = nn.Linear(self.features, 1)
        self.time = nn.Linear(self.output, self.output)

        #self.time = None
        self.tanh_va = nn.Tanh()
        self.flatten = nn.Flatten()      
        #self.init_weights()
        self.unfreeze_last_n_blocks(4)
                
    def freeze_conv_only(self):
        for param in self.wav_model.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, num_blocks: int) :
        for param in self.wav_model.parameters():
            param.requires_grad = False

        for i in range(0, num_blocks):
            for param in self.wav_model.encoder.layers[-1 * (i + 1)].parameters():
                param.requires_grad = True

    def forward(self, input_values):
        x, _ = self.wav_model(input_values)
      

        x = self.transformer_layer(x)        
        x = x.permute(0, 2, 1)       
        x = self.time_downsample(x)  
        x = x.permute(0, 2, 1)     
        x = self.feature_downsample(x)
        x = self.flatten(x)
        if self.time == None:
            self.time = nn.Linear(x.shape[-1], self.output).to("cuda")
        x = self.time(x)
        x = self.tanh_va(x)
        return x
    
    
class RespBertAttionModelV2(Wav2Vec2PreTrainedModel):
    def __init__(self, bert_config, config):
        super().__init__(bert_config)
        self.config = bert_config
        self.output = config['output_size']
        
        if bert_config.model_type == "wav2vec2":
            self.wav_model = Wav2Vec2Model(bert_config)
        elif bert_config.model_type == "hubert":
            self.wav_model = HubertModel(bert_config)
        elif bert_config.model_type == "wavlm":
            self.wav_model = WavLMModel(bert_config)
        else:
            raise ValueError("Unsupported model type")
        self.wav_model.encoder.layers = self.wav_model.encoder.layers[7] 

        self.d_model = bert_config.hidden_size
        self.features = config['hidden_units']
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=12, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=config['n_attion'])

        self.num_cnn_layers = 3
        self.convs = nn.ModuleList([
            nn.Conv1d(self.d_model if i == 0 else self.features, self.features, kernel_size=3, padding=1)
            for i in range(self.num_cnn_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.features) for _ in range(self.num_cnn_layers)])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.output)


        self.feature_downsample = nn.Linear(self.features, 1)
        self.time = None
        self.tanh_va = nn.Tanh()
        self.flatten = nn.Flatten()      
        #self.init_weights()
        self.unfreeze_last_n_blocks(1)
                
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
        x = self.wav_model(input_values)
        print(x.shape)       
        x = self.transformer_layer(x)        
        x = x.permute(0, 2, 1)       

        for conv, bn in zip(self.convs, self.bns):

            x = conv(x)
            residual = x
            x = bn(x)
            x = self.gelu(x)
            x = self.dropout(x)
            x += residual  # Skip connection
        x= self.adaptive_pool(x)

        x = x.permute(0, 2, 1) 
        x = self.feature_downsample(x)
        x = self.flatten(x)
        if self.time == None:
            self.time = nn.Linear(self.output, self.output).to("cuda")
        x = self.time(x)
        x = self.tanh_va(x)
        return x
