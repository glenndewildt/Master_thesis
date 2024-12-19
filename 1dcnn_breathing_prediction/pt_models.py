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
    AutoFeatureExtractor,
    AutoConfig,
    AutoModelForAudioClassification,
    AutoModel,
    AutoProcessor,
    Wav2Vec2FeatureExtractor
)
from transformers import AutoProcessor, AutoModelForCTC

import math
import json
#from flash_attn import flash_attn_func
from pt_utils import *
from pt_dataset import *

import torch
import torch.nn as nn
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.init import xavier_uniform_, kaiming_uniform_
from tqdm import tqdm
class WavlmCNNModel(nn.Module):
    def __init__(self, config):
        super(WavlmCNNModel, self).__init__()
        self.output = config['output_size']
        
        self.wav_model = AutoModel.from_pretrained(config["model_name"])
        self.d_model = self.wav_model.config.hidden_size       
        self.features = config['hidden_units']
        
        self.cnn_down = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),  
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(self.d_model, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(), 
            nn.Dropout(0.2),         
        )
        
        self.feature_down = nn.Sequential(
            nn.Linear(self.features, 1),
            nn.Flatten()
        )
        self.time_down = nn.Sequential(
            nn.Linear(1499, self.output),
            nn.Tanh()  # Ensures output is between -1 and 1
        )
        self.flatten = nn.Flatten()      
        
        self._initialize_weights()
        self.freeze_all_layers()
                
    def _initialize_weights(self):
        # Initialize weights for CNN layers
        for m in self.cnn_down:
            if isinstance(m, nn.Conv1d):
                kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        # Initialize weights for Linear layers
        for m in self.feature_down:
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                m.bias.data.zero_()
        for m in self.time_down:
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                m.bias.data.zero_()



    def freeze_all_layers(self):
        for param in self.wav_model.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, num_blocks: int):
        for param in self.wav_model.parameters():
            param.requires_grad = False

        for i in range(0, num_blocks):
            for param in self.wav_model.encoder.layers[-1 * (i + 1)].parameters():
                param.requires_grad = True

    def forward(self, input_values):
        x = self.wav_model(**input_values)[0]
        x = x.permute(0, 2, 1)       
        x = self.cnn_down(x)  
        x = x.permute(0, 2, 1)
        x = self.feature_down(x)
        x = self.time_down(x)
        return x
class RespBertAttention(nn.Module):
    def __init__(self, config):
        super(RespBertAttention, self).__init__()
        self.output = config['output_size']
        
        self.wav_model = AutoModel.from_pretrained(config["model_name"])
        self.d_model = self.wav_model.config.hidden_size       
        self.features = config['hidden_units']
        
        # Transformer Encoder with Layer Normalization and Residual Connections
        self.transformer_layer = TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=16, 
            dropout=0.2, 
        )
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=config['n_attion'])

        self.cnn_down = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),  
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(self.d_model, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(), 
            nn.Dropout(0.2),         
        )
        
        self.feature_down = nn.Sequential(
            nn.Linear(self.features, 1),
            nn.Flatten()
        )
        self.time_down = nn.Sequential(
            nn.Linear(1499, self.output),
            nn.Tanh()  # Ensures output is between -1 and 1
        )
        self.flatten = nn.Flatten()      
        
        self.freeze_all_layers()
                

    def freeze_all_layers(self):
        for param in self.wav_model.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, num_blocks: int):
        for param in self.wav_model.parameters():
            param.requires_grad = False

        for i in range(0, num_blocks):
            for param in self.wav_model.encoder.layers[-1 * (i + 1)].parameters():
                param.requires_grad = True

    def forward(self, input_values):
        x = self.wav_model(**input_values)[0]
        x = self.transformer_encoder(x)        

        x = x.permute(0, 2, 1)       
        x = self.cnn_down(x)  
        x = x.permute(0, 2, 1)
  
        x = self.feature_down(x)

        x = self.time_down(x)

        return x

##BASED IN APPLE PAPER
## question1 : I use only the last layer from the lstm, is this a good assumtion?
## they say they make use of a ebedding layer with 128 so did add that before the final layer but also not sure they ment that this way 
class Wav2Vec2ConvLSTMModel(nn.Module):
    def __init__(self, config):
        super(Wav2Vec2ConvLSTMModel, self).__init__()
        self.wav_model = AutoModel.from_pretrained(config["model_name"])
        self.wav_model.encoder.layers = self.wav_model.encoder.layers[:6]
        # Freeze the Wav2Vec2 model's parameters
        for param in self.wav_model.parameters():
            param.requires_grad = False

        self.input_features = self.wav_model.config.hidden_size       
        self.conv = nn.Conv1d(in_channels=self.input_features,
                              out_channels=self.input_features,
                              kernel_size=3,
                              padding=1, dilation = 1)
               
        self.lstm = nn.LSTM(input_size= 2* self.input_features,
                            hidden_size=config['hidden_units'],
                            num_layers=config['n_lstm'],
                            batch_first=True)
        self.embedding = nn.Linear(config['hidden_units'] * 1499, config['hidden_units'])
        self.output = nn.Linear(config['hidden_units'], config['output_size'])
        self.flatten = nn.Flatten()
        
    

    def forward(self, input_values):
        with torch.no_grad():
            wav2vec_features=  self.wav_model(**input_values).last_hidden_state
        x = wav2vec_features.permute(0, 2, 1)         
        conv_features = self.conv(x) # goes finds patterns in the features over all for breathing features for each timestep      
        conv_features = conv_features.permute(0, 2, 1)
        concat_features = torch.concat([wav2vec_features,conv_features], dim=-1)      
        lstm_out, _ = self.lstm(concat_features) # for each time step there are now 128 features into a lstm       
        #last_time_step = lstm_out[:, -1, :]  # get the lest timestep to get the the timestep with all the incorparated data from the other steps (hopfully) 
        flattend_lstm = self.flatten(lstm_out)
        embed = self.embedding(flattend_lstm)   # a linear layer that goes from a flatted ouput of all the states of the lstm with a feature size of 128 to an embedding layer of 128 
        output = self.output(embed)    # last layer goes from 128 from the embedding layer to (window_size * sample rate) in this case it is 30 seconds of the window and 25 datapoints per second for the beathing signal
        
        return output
    
##BASED ON VRB HARMA2023 PAPER 
## Question1 : Again do i only use the last layer of the GRU, it has a output of 64 so seems a little small, the output has 400 as output shape i would say that for each timestep but could not find it   
## Question2 : there was not descriped a activation function, and a dense layer for the output.
class VRBModel(nn.Module):
    def __init__(self,config):
        super(VRBModel, self).__init__()
        self.wav_model = AutoModel.from_pretrained(config["model_name"])
        # Freeze the Wav2Vec2 model's parameters
        for param in self.wav_model.parameters():
            param.requires_grad = False
        self.input_features = self.wav_model.config.hidden_size       
        self.lstm = nn.LSTM(input_size=self.input_features,
                          hidden_size=config['hidden_units'],
                          num_layers=config['n_gru'],
                          batch_first=True)
        self.fc = nn.Linear(config['hidden_units'] * 1499, config['output_size'])
        self.flatten = nn.Flatten()
      
    
    def forward(self, input_values):
        with torch.no_grad():
            wav_output = self.wav_model(**input_values).last_hidden_state
        lstm_out, _ = self.lstm(wav_output)  
            
        #last_time_step = gru_out[:, -1, :]
        flattend_lstm = self.flatten(lstm_out)
     
        fc = self.fc(flattend_lstm)
        
        return fc
    
class WavlmVRBModel(nn.Module):
    def __init__(self,config):
        super(WavlmVRBModel, self).__init__()
        self.wav_model = AutoModel.from_pretrained(config["model_name"])
        # Freeze the Wav2Vec2 model's parameters
        for param in self.wav_model.parameters():
            param.requires_grad = False
        self.input_features = self.wav_model.config.hidden_size       
        self.lstm = nn.LSTM(input_size=self.input_features,
                          hidden_size=config['hidden_units'],
                          num_layers=config['n_gru'],
                          batch_first=True)
        self.fc = nn.Linear(config['hidden_units'] * 1999, config['output_size'])
        self.flatten = nn.Flatten()
      
    
    def forward(self, input_values):
        with torch.no_grad():
            wav_output = self.wav_model(**input_values).last_hidden_state
        #print(wav_output.shape)
        lstm_out, _ = self.lstm(wav_output)  
            
        #last_time_step = gru_out[:, -1, :]
        flattend_lstm = self.flatten(lstm_out)
     
        fc = self.fc(flattend_lstm)
        
        return fc
    

## MY PROPOSED MODEL DESIGNS
class RespBertLSTM(nn.Module):
    def __init__(self,config):
        super(RespBertLSTM, self).__init__()
        self.output = config['output_size']
        
        self.wav_model = AutoModel.from_pretrained(config["model_name"])

        self.input_features = self.wav_model.config.hidden_size       
        
        self.features = config['hidden_units']
        
        self.lstm = nn.LSTM(input_size=self.input_features, hidden_size=self.features, num_layers=config['n_lstm'], batch_first=True, dropout=0.2, bidirectional=False)
        
        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.features , self.features , kernel_size=3, padding="same"),
            nn.BatchNorm1d(self.features ),  
            nn.GELU(),
            nn.Dropout(0.2),         

            nn.Conv1d(self.features , self.features , kernel_size=3, padding="same"),
            nn.BatchNorm1d(self.features ),  
            nn.GELU(),
            nn.Dropout(0.2),         
  
        )

        self.time = nn.Linear(1499, self.output)
        self.feature_downsample = nn.Linear(self.features, 1)
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

        x= self.wav_model(**input_values)[0]
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

class RespBertAttionModel(nn.Module):
    def __init__(self, config):
        super(RespBertAttionModel, self).__init__()
        self.output = config['output_size']
        
        self.wav_model = AutoModel.from_pretrained(config["model_name"])

        #self.wav_model.encoder.layers = self.wav_model.encoder.layers[:]

        self.d_model = self.wav_model.config.hidden_size       
        
        self.features = config['hidden_units']
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=16, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=config['n_attion'])

        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),  
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Conv1d(self.d_model, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(), 
            nn.Dropout(0.2),       
            #nn.AdaptiveAvgPool1d(self.output),
  
        )
        self.time = nn.Linear(1499, self.output)
        self.feature_downsample = nn.Linear(self.features, 1)
        

        #self.time = None
        self.tanh_va = nn.Tanh()
        self.flatten = nn.Flatten()      
        #self.init_weights()
        self.unfreeze_last_n_blocks(8)
                
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
        x= self.wav_model(**input_values)[0]
      

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
    
class RespBertCNNModel(nn.Module):
    def __init__(self, config):
        super(RespBertCNNModel, self).__init__()
        self.output = config['output_size']
        
        self.wav_model = AutoModel.from_pretrained(config["model_name"])

        #self.wav_model.encoder.layers = self.wav_model.encoder.layers[:]

        self.d_model = self.wav_model.config.hidden_size       
        
        self.features = config['hidden_units']

        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),  
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Conv1d(self.d_model, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(), 
            nn.Dropout(0.2),       
            #nn.AdaptiveAvgPool1d(self.output),
  
        )
        self.time = nn.Linear(1499, self.output)
        self.feature_downsample = nn.Linear(self.features, 1)
        

        #self.time = None
        self.tanh_va = nn.Tanh()
        self.flatten = nn.Flatten()      
        #self.init_weights()
        self.unfreeze_last_n_blocks(8)

        self.set_dropout(layers = 4 ,dropout=0.2)
        self.set_dropout(layers = 2 ,dropout=0.3)
        #self.wav_model.gradient_checkpointing_enable()
        
    def set_dropout(self, dropout = 0.2, layers = 8):
        num_layers = len(self.wav_model.encoder.layers)  # Total number of transformer layers
        for i in range(num_layers - layers, num_layers):
                # Access the dropout layers within the transformer block
            layer = self.wav_model.encoder.layers[i]  
            layer.attention.dropout = dropout# Modify attention dropout
            layer.dropout.p = dropout  # Modify hidden/output dropout    
            
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
        x= self.wav_model(**input_values)[0]
      

        #x = self.transformer_layer(x)        
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
class RespBertCNNModel(nn.Module):
    def __init__(self, config):
        super(RespBertCNNModel, self).__init__()
        self.output = config['output_size']
        
        self.wav_model = AutoModel.from_pretrained(config["model_name"])

        #self.wav_model.encoder.layers = self.wav_model.encoder.layers[:]

        self.d_model = self.wav_model.config.hidden_size       
        
        self.features = config['hidden_units']

        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),  
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Conv1d(self.d_model, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(), 
            nn.Dropout(0.2),       
            #nn.AdaptiveAvgPool1d(self.output),
  
        )
        self.time = nn.Linear(1499, self.output)
        self.feature_downsample = nn.Linear(self.features, 1)
        

        #self.time = None
        self.tanh_va = nn.Tanh()
        self.flatten = nn.Flatten()      
        #self.init_weights()
        self.unfreeze_last_n_blocks(8)

        self.set_dropout(layers = 4 ,dropout=0.2)
        self.set_dropout(layers = 2 ,dropout=0.3)
        #self.wav_model.gradient_checkpointing_enable()
        
    def set_dropout(self, dropout = 0.2, layers = 8):
        num_layers = len(self.wav_model.encoder.layers)  # Total number of transformer layers
        for i in range(num_layers - layers, num_layers):
                # Access the dropout layers within the transformer block
            layer = self.wav_model.encoder.layers[i]  
            layer.attention.dropout = dropout# Modify attention dropout
            layer.dropout.p = dropout  # Modify hidden/output dropout    
            
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
        x= self.wav_model(**input_values)[0]
      

        #x = self.transformer_layer(x)        
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
              
#### TEST MODEL    
class RespBertLSTMModelTEST(nn.Module):
    def __init__(self, config):
        super(RespBertLSTMModelTEST, self).__init__()
        self.output = config['output_size']
        self.wav_model = AutoModel.from_pretrained(config["model_name"])
        self.wav_model.encoder.layers = self.wav_model.encoder.layers[0:10]
        self.input_features = self.wav_model.config.hidden_size       
        self.features = config['hidden_units']
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.features, 
                            num_layers=config['n_lstm'], batch_first=True, dropout=0.2)
        
        # Time downsample layer
        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.features, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(self.output),
        )
        
        # Final layers
        self.time = nn.Linear(self.output, self.output)
        self.feature_downsample = nn.Linear(self.features, 1)
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
        ## input is attention mask and audio (batch, data{audio, attaionmask})
        for key, value in input_values.items():
            print(f"Shape of input with key '{key}':", value.shape)

        # Step 1: WavLM model
        # Input: (batch_size, audio_length)
        # Output: (batch_size, sequence_length, hidden_size) the first output [0] is the output from trensformerlayers and the second from the feature extraction of the cnn layers
        # the seqnce lenght is based on a step size of 20ms so if you have 30 seconds is 30/0.002 - 1500.
        # The WavLM model processes the raw audio input and returns contextualized representations
        hubert_ouput = self.wav_model(**input_values)
        for key, value in hubert_ouput.items():
            print(f"Shape of WavLM model output with key '{key}':", value.shape)
        
        encoder_output = hubert_ouput[0]
        print(f"Getting the trensformer output form WavLM output: {encoder_output.shape}")


        # Step 2: LSTM layer
        # Input: (batch_size, sequence_length, hidden_size) 
        # Output: (batch_size, sequence_length, features)
        # The LSTM processes the sequence, maintaining the time dimension
        # so for each time step the lstm goes over the feature size of the model in this case 1024 and creates hidden layer 256 features. the next layer gets the information form the last layer hidden state
        hidden_lstm_outputs, _ = self.lstm(encoder_output)
        print(f"After LSTM get all hidden states: {hidden_lstm_outputs.shape}")

        # Step 3: Permute dimensions for Conv1d
        # Input: (batch_size, sequence_length, features)
        # Output: (batch_size, features, sequence_length)
        # Rearrange dimensions to apply 1D convolution over time
        hidden_lstm_outputs = hidden_lstm_outputs.permute(0, 2, 1)
        print(f"After permute: {hidden_lstm_outputs.shape}")

        # Step 4: Time downsample
        # Input: (batch_size, features, sequence_length)
        # Output: (batch_size, features, output_size)
        # Reduce the time dimension to a fixed size averga pooling to (output_size) and for finding patterns in the features from the lstm in the time demetion. 
        
        features_per_time = self.time_downsample(hidden_lstm_outputs)
        print(f"After time downsample: {features_per_time.shape}")

        # Step 5: Permute back
        # Input: (batch_size, features, output_size)
        # Output: (batch_size, output_size, features)
        # Rearrange dimensions for the linear layer
        features_per_time = features_per_time.permute(0, 2, 1)
        print(f"After permute back: {features_per_time.shape}")

        # Step 6: Feature downsample
        # Input: (batch_size, output_size, features)
        # Output: (batch_size, output_size, 1)
        # Reduce the feature dimension to 1 so for each timestep a linear layer is applied to have one demetion for each demention. afther this the demention should be (batch, 800, 1)
        one_feature_per_timestep = self.feature_downsample(features_per_time)
        print(f"After feature downsample: {one_feature_per_timestep.shape}")

        # Step 7: Flatten
        # Input: (batch_size, output_size, 1)
        # Output: (batch_size, output_size)
        # Remove the last dimension
        flattend_output = self.flatten(one_feature_per_timestep)
        print(f"After flatten: {flattend_output.shape}")

        # Step 8: Final linear layer
        # Input: (batch_size, output_size)
        # Output: (batch_size, output_size)
        # Apply final transformation this to give the model the option to look in the time demention and see which one are inmportnat for the prediction.
        fc_output = self.time(flattend_output)
        print(f"After final linear: {fc_output.shape}")

        # Step 9: Tanh activation
        # Input: (batch_size, output_size)
        # Output: (batch_size, output_size)
        # Apply tanh activation to constrain values between -1 and 1
        final_output = self.tanh_va(fc_output)
        print(f"Final output: {final_output.shape}")

        return final_output
    
class RespBertCNNModel_skip(nn.Module):
    def __init__(self, config):
        super(RespBertCNNModel_skip, self).__init__()
        self.output = config['output_size']
        
        self.wav_model = AutoModel.from_pretrained(config["model_name"])

        self.d_model = self.wav_model.config.hidden_size       
        
        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),  
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),  
            nn.GELU(), 
            nn.Dropout(0.3),

        )
        
        self.lstm = nn.LSTM(input_size= 2* self.d_model,
                            hidden_size=config['hidden_units'],
                            num_layers=2,
                            dropout=0.3,
                            batch_first=True)
        
        self.time = nn.Linear(1999 * config['hidden_units'], self.output)
        

        #self.time = None
        self.tanh_va = nn.Tanh()
        self.flatten = nn.Flatten()      
        #self.init_weights()
        #self.unfreeze_last_n_blocks(8)

        #self.set_dropout(layers = 4 ,dropout=0.3)
        #self.wav_model.gradient_checkpointing_enable()
        
    def set_dropout(self, dropout = 0.2, layers = 8):
        num_layers = len(self.wav_model.encoder.layers)  # Total number of transformer layers
        for i in range(num_layers - layers, num_layers):
                # Access the dropout layers within the transformer block
            layer = self.wav_model.encoder.layers[i]  
            layer.attention.dropout = dropout# Modify attention dropout
            layer.dropout.p = dropout  # Modify hidden/output dropout    
            
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
        with torch.no_grad():
            features = self.wav_model(**input_values)[0]
        x = features.permute(0, 2, 1)       
        cnn_features = self.time_downsample(x)  
        cnn_features = cnn_features.permute(0, 2, 1)
        concat_values = torch.concat([features, cnn_features],dim = -1)  
        lstm, _ = self.lstm(concat_values)
        x = self.flatten(lstm)
        x = self.time(x)
        x = self.tanh_va(x)
        return x  
    
class WavLMCNNLSTM(nn.Module):
    def __init__(self, config):
        super(WavLMCNNLSTM, self).__init__()
        self.wav_model = AutoModel.from_pretrained(config["model_name"])

        self.input_features = self.wav_model.config.hidden_size       
        self.conv = nn.Conv1d(in_channels=self.input_features,
                                out_channels=self.input_features,
                                kernel_size=3,
                                padding=1, dilation = 1)
                
        self.lstm = nn.LSTM(input_size= 2* self.input_features,
                            hidden_size=config['hidden_units'],
                            num_layers=config['n_lstm'],
                            batch_first=True)
        self.embedding = nn.Linear(config['hidden_units'] * 1499, config['output_size'])
        self.output = nn.Linear(config['hidden_units'], config['output_size'])
        self.flatten = nn.Flatten()
        self.unfreeze_last_n_blocks(18)

        
    def unfreeze_last_n_blocks(self, num_blocks: int) :
        for param in self.wav_model.parameters():
            param.requires_grad = False

        for i in range(0, num_blocks):
            for param in self.wav_model.encoder.layers[-1 * (i + 1)].parameters():
                param.requires_grad = True

    def forward(self, input_values):
        wav2vec_features=  self.wav_model(**input_values).last_hidden_state
        x = wav2vec_features.permute(0, 2, 1)         
        conv_features = self.conv(x) # goes finds patterns in the features over all for breathing features for each timestep      
        conv_features = conv_features.permute(0, 2, 1)
        concat_features = torch.concat([wav2vec_features,conv_features], dim=-1)      
        lstm_out, _ = self.lstm(concat_features) # for each time step there are now 128 features into a lstm       
        #last_time_step = lstm_out[:, -1, :]  # get the lest timestep to get the the timestep with all the incorparated data from the other steps (hopfully) 
        flattend_lstm = self.flatten(lstm_out)
        embed = self.embedding(flattend_lstm)   # a linear layer that goes from a flatted ouput of all the states of the lstm with a feature size of 128 to an embedding layer of 128 
        #output = self.output(embed)    # last layer goes from 128 from the embedding layer to (window_size * sample rate) in this case it is 30 seconds of the window and 25 datapoints per second for the beathing signal
        
        return embed
    
class RespBertCNNModelV2(nn.Module):
    def __init__(self, config):
        super(RespBertCNNModelV2, self).__init__()
        self.output = config['output_size']
        
        self.wav_model = AutoModel.from_pretrained(config["model_name"])

        #self.wav_model.encoder.layers = self.wav_model.encoder.layers[:]

        self.d_model = self.wav_model.config.hidden_size       
        
        self.features = config['hidden_units']

        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),  
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Conv1d(self.d_model, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(), 
            nn.Dropout(0.2),       
            #nn.AdaptiveAvgPool1d(self.output),
  
        )
        self.time = nn.Linear(self.features*1499, self.output)
        self.feature_downsample = nn.Linear(self.features, 1)
        

        #self.time = None
        self.tanh_va = nn.Tanh()
        self.flatten = nn.Flatten()      
        #self.init_weights()
        self.unfreeze_last_n_blocks(8)

        self.set_dropout(layers = 4 ,dropout=0.2)
        self.set_dropout(layers = 2 ,dropout=0.3)
        #self.wav_model.gradient_checkpointing_enable()
        
    def set_dropout(self, dropout = 0.2, layers = 8):
        num_layers = len(self.wav_model.encoder.layers)  # Total number of transformer layers
        for i in range(num_layers - layers, num_layers):
                # Access the dropout layers within the transformer block
            layer = self.wav_model.encoder.layers[i]  
            layer.attention.dropout = dropout# Modify attention dropout
            layer.dropout.p = dropout  # Modify hidden/output dropout    
            
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
        x= self.wav_model(**input_values)[0]
      

        #x = self.transformer_layer(x)        
        x = x.permute(0, 2, 1)       
        x = self.time_downsample(x)  
        x = x.permute(0, 2, 1)     
        x = self.flatten(x)
        x = self.time(x)
        return x

# MY PROPOSED MODEL DESIGNS
class RespBertLSTMModelV2(nn.Module):
    def __init__(self,config):
        super(RespBertLSTMModel, self).__init__()
        self.output = config['output_size']
        
        self.wav_model = AutoModel.from_pretrained(config["model_name"])

        self.input_features = self.wav_model.config.hidden_size       
        
        self.features = config['hidden_units']
        
        self.lstm = nn.LSTM(input_size=self.input_features, hidden_size=self.features, num_layers=config['n_lstm'], batch_first=True, dropout=0.2, bidirectional=False)
        
        self.time = nn.Linear(1499*self.input_features, self.output)
        self.flatten = nn.Flatten()
        
        self.unfreeze_last_n_blocks(16)
                
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
        x= self.wav_model(**input_values)[0]
        x, _ = self.lstm(x)       
        x = self.flatten(x)
        x = self.time(x)

        return x
    
import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class RespBertLSTMCNNTransformerModel(nn.Module):
    def __init__(self, config):
        super(RespBertLSTMCNNTransformerModel, self).__init__()
        self.output = config['output_size']
        self.wav_model = AutoModel.from_pretrained(config["model_name"])
        self.input_features = self.wav_model.config.hidden_size
        self.features = config['hidden_units']

        # LSTM layers
        self.lstm = nn.LSTM(input_size=self.input_features, hidden_size=self.features, num_layers=config['n_lstm'], batch_first=True, dropout=0.2, bidirectional=False)

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_features, self.input_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.input_features),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(self.input_features, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Linear projection to match feature sizes
        self.projection = nn.Linear(self.input_features, self.features)

        # Transformer layers
        encoder_layer = TransformerEncoderLayer(d_model=self.features * 3, nhead=8, dropout=0.2)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1)

        self.fc = nn.Linear((1499*self.features * 3) , self.output)
        self.flatten = nn.Flatten()

        self.unfreeze_last_n_blocks(8)

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
        wav = self.wav_model(**input_values)[0]
        cnn_output = self.cnn(wav.permute(0, 2, 1))
        lstm_output, _ = self.lstm(wav)

        # Project Wav2Vec2 output to match CNN and LSTM feature size
        wav_projected = self.projection(wav)

        # Combine CNN, LSTM, and projected Wav2Vec2 outputs
        combined_output = torch.cat((cnn_output.permute(0, 2, 1), lstm_output, wav_projected), dim=2)
        combined_output = combined_output.permute(1, 0, 2)  # (batch, sequence, features)

        # Pass combined output through Transformer layers
        transformer_output = self.transformer(combined_output)
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch, sequence, features)

        
        # Final fully connected layer
        output = self.fc(self.flatten(transformer_output))

        return output
class WALMLSTM(nn.Module):
    def __init__(self, config):
        super(WALMLSTM, self).__init__()
        self.output = config['output_size']
        self.wav_model = AutoModel.from_pretrained(config["model_name"])

        self.input_features = self.wav_model.config.hidden_size
        self.features = config['hidden_units']

        # LSTM layers
        self.lstm = nn.LSTM(input_size=self.input_features, hidden_size=self.features, num_layers=config['n_lstm'], batch_first=True, dropout=0.2, bidirectional=True)

        self.fc = nn.Linear((1499*self.features * 2) , self.output)
        self.flatten = nn.Flatten()

        self.unfreeze_last_n_blocks(16)

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
        wav = self.wav_model(**input_values)[0]
        lstm_output, _ = self.lstm(wav)

        
        # Final fully connected layer
        output = self.fc(self.flatten(lstm_output))

        return output
    
class RespBertCNN_12_Model(nn.Module):
    def __init__(self, config):
        super(RespBertCNN_12_Model, self).__init__()
        self.output = config['output_size']
        
        self.wav_model = AutoModel.from_pretrained(config["model_name"])
        

        #self.wav_model.encoder.layers = self.wav_model.encoder.layers[:12]

        self.d_model = self.wav_model.config.hidden_size       
        
        self.features = config['hidden_units']

        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),  
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Conv1d(self.d_model, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(), 
            nn.Dropout(0.2),       
            #nn.AdaptiveAvgPool1d(self.output),
  
        )
        self.time = nn.Linear(1999, self.output)
        self.feature_downsample = nn.Linear(self.features, 1)
        

        #self.time = None
        self.tanh_va = nn.Tanh()
        self.flatten = nn.Flatten()      
        #self.init_weights()
        self.unfreeze_last_n_blocks(12)

        self.set_dropout(layers = 8 ,dropout=0.2)
        #self.wav_model.gradient_checkpointing_enable()
        
    def set_dropout(self, dropout = 0.2, layers = 8):
        num_layers = len(self.wav_model.encoder.layers)  # Total number of transformer layers
        for i in range(num_layers - layers, num_layers):
                # Access the dropout layers within the transformer block
            layer = self.wav_model.encoder.layers[i]  
            layer.attention.dropout = dropout# Modify attention dropout
            layer.dropout.p = dropout  # Modify hidden/output dropout    
            
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
        x= self.wav_model(**input_values)[0]
      

        #x = self.transformer_layer(x)        
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
    
class HuBertCNN_con_Model(nn.Module):
    def __init__(self, config):
        super(HuBertCNN_con_Model, self).__init__()
        self.output = config['output_size']
        
        self.wav_model = AutoModel.from_pretrained(config["model_name"],attn_implementation="flash_attention_2")
        

        #self.wav_model.encoder.layers = self.wav_model.encoder.layers[:12]

        self.d_model = self.wav_model.config.hidden_size       
        
        self.features = config['hidden_units']

        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),  
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),  
            nn.GELU(), 
            nn.Dropout(0.2),       
            #nn.AdaptiveAvgPool1d(self.output),
  
        )
        self.time = nn.Linear(1499, self.output)
        self.feature_downsample = nn.Linear(self.d_model*2, 1)
       

        #self.time = None
        self.tanh_va = nn.Tanh()
        self.flatten = nn.Flatten()      
        #self.init_weights()
        self.unfreeze_last_n_blocks(16)

        self.set_dropout(layers = 8 ,dropout=0.2)
        #self.wav_model.gradient_checkpointing_enable()
        
    def set_dropout(self, dropout = 0.2, layers = 8):
        num_layers = len(self.wav_model.encoder.layers)  # Total number of transformer layers
        for i in range(num_layers - layers, num_layers):
                # Access the dropout layers within the transformer block
            layer = self.wav_model.encoder.layers[i]  
            layer.attention.dropout = dropout# Modify attention dropout
            layer.dropout.p = dropout  # Modify hidden/output dropout    
            
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
        outputs = self.wav_model(**input_values, output_hidden_states=True)
        cnn = outputs.hidden_states[0] # Shape: (batch_size, sequence_length, hidden_size)
        trans = outputs.hidden_states[-1] # Shape: (batch_size, sequence_length, hidden_size)

        #x = self.transformer_layer(x)        
        trans = trans.permute(0, 2, 1)       
        trans = self.time_downsample(trans)  
        trans = trans.permute(0, 2, 1)
        

        concat_features = torch.concat([trans,cnn], dim=-1)      
        x = self.feature_downsample(concat_features)

        x = self.flatten(x)
        if self.time == None:
            self.time = nn.Linear(x.shape[-1], self.output).to("cuda")
        x = self.time(x)
        #x = self.tanh_va(x)
        return x
class RespBertCNN(nn.Module):
    def __init__(self, config):
        super(RespBertCNN, self).__init__()
        self.output = config['output_size']
        
        self.wav_model = AutoModel.from_pretrained(config["model_name"])
    
        self.d_model = self.wav_model.config.hidden_size       
        
        self.features = config['hidden_units']

        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),  
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Conv1d(self.d_model, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(), 
            nn.Dropout(0.2),         
        )
        
        self.time = nn.Linear(1499, self.output)
        self.feature_downsample = nn.Linear(self.features, 1)
        self.flatten = nn.Flatten()      
        self.unfreeze_last_n_blocks(config['number_finetune'])

        self.set_dropout(layers = 2 ,dropout=0.2)
        
    def set_dropout(self, dropout = 0.2, layers = 8):
        num_layers = len(self.wav_model.encoder.layers)  # Total number of transformer layers
        for i in range(num_layers - layers, num_layers):
                # Access the dropout layers within the transformer block
            layer = self.wav_model.encoder.layers[i]  
            layer.attention.dropout = dropout# Modify attention dropout
            layer.dropout.p = dropout  # Modify hidden/output dropout    
            
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
        x= self.wav_model(**input_values)[0]
        x = x.permute(0, 2, 1)       
        x = self.time_downsample(x)  
        x = x.permute(0, 2, 1)     
        x = self.feature_downsample(x)
        x = self.flatten(x)
        x = self.time(x)
        return x


class HuBertAttention_12_OG_Model(nn.Module):
    def __init__(self, config):
        super(HuBertAttention_12_OG_Model, self).__init__()
        self.output = config['output_size']
                
        self.features = config['hidden_units']
        
        self.wav_model = AutoModel.from_pretrained(config["model_name"],attn_implementation="flash_attention_2")
        self.d_model = self.wav_model.config.hidden_size       

        # Transformer Encoder with Layer Normalization and Residual Connections
        self.transformer_layer = TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=16, 
            dropout=0.2, 
        )
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=2)
        #self.wav_model.encoder.layers = self.wav_model.encoder.layers[:12]


        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),  
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Conv1d(self.d_model, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),  
            nn.GELU(), 
            nn.Dropout(0.2),       
            nn.AdaptiveAvgPool1d(self.output),
  
        )
        self.feature_downsample = nn.Linear(self.features, 1)
        

        #self.time = None
        self.tanh_va = nn.Tanh()
        self.flatten = nn.Flatten()      
        #self.init_weights()
        self.unfreeze_last_n_blocks(12)

        self.set_dropout(layers = 8 ,dropout=0.2)
        #self.wav_model.gradient_checkpointing_enable()
        
    def set_dropout(self, dropout = 0.2, layers = 8):
        num_layers = len(self.wav_model.encoder.layers)  # Total number of transformer layers
        for i in range(num_layers - layers, num_layers):
                # Access the dropout layers within the transformer block
            layer = self.wav_model.encoder.layers[i]  
            layer.attention.dropout = dropout# Modify attention dropout
            layer.dropout.p = dropout  # Modify hidden/output dropout    
            
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
        x= self.wav_model(**input_values)[0]
        x = self.transformer_layer(x)     
           
        x = x.permute(0, 2, 1)       
        x = self.time_downsample(x)  
        x = x.permute(0, 2, 1)     
        x = self.feature_downsample(x)
        x = self.flatten(x)
        x = self.tanh_va(x)
        return x
