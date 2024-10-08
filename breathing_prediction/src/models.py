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
import math
import json
#from flash_attn import flash_attn_func
from flash_attn.bert_padding import unpad_input, pad_input

from flash_attn.modules.mha import FlashSelfAttention
from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)
from typing import Optional, Tuple

class FlashAttention2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = config.attention_dropout
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.has_relative_position_bias = True
        if self.has_relative_position_bias:
            self.rel_attn_embed = nn.Embedding(config.num_buckets, self.num_heads)    
    def _compute_position_bias(self, seq_length, device):
        # Simplified position bias computation
        position_bias = self.rel_attn_embed.weight.unsqueeze(1).repeat(1, seq_length, 1)
        return position_bias.permute(2, 0, 1).unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        bsz, seq_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states) * self.scaling
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.has_relative_position_bias and position_bias is None:
            position_bias = self._compute_position_bias(seq_len, hidden_states.device)

        if attention_mask is not None:
            attention_mask = attention_mask.view(bsz, seq_len)
            q, indices, cu_seqlens, max_seqlen = unpad_input(q, attention_mask)
            k, _, _, _ = unpad_input(k, attention_mask)
            v, _, _, _ = unpad_input(v, attention_mask)

            output = flash_attn_func(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                causal=False, 
                softmax_scale=None,
                return_attn_probs=output_attentions
            )

            output = pad_input(output, indices, bsz, seq_len)
        else:
            output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,
                softmax_scale=None,
                return_attn_probs=output_attentions
            )

        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        output = self.out_proj(output)
        outputs = (hidden_states, attention_mask, output_attentions)


        return outputs

class WavLMWithFlashAttention2(WavLMModel):
    def __init__(self, config):
        super().__init__(config)

        # Replace the attention layers with Flash Attention 2
        for layer in self.encoder.layers:
            layer.attention = FlashAttention2(config)

def load_wavlm_with_flash_attention2(model_name_or_path):
    original_model = WavLMModel.from_pretrained(model_name_or_path)
    config = original_model.config
    # Replace the attention layers with Flash Attention 2
    for layer in original_model.encoder.layers:
        layer.attention = FlashAttention2(config)
    #flash_model = WavLMWithFlashAttention2(config)
    #flash_model.load_state_dict(original_model.state_dict(), strict=False)
    #del original_model
    return original_model


def convert_all_layers_to_flash_attention(model):
    """
    Convert all transformer layers in the given model to use FlashAttention 2.

    Args:
    - model (nn.Module): The pre-trained model with transformer layers to convert.
    
    Returns:
    - model (nn.Module): The model with FlashAttention applied.
    """
    #print(model)
    
    def replace_attention_with_flash(layer):
        """
        Recursively replace all instances of Huggingface's attention with FlashAttention.
        """
        for name, submodule in layer.named_children():
            # Huggingface attention layers are usually named `self_attn`, `attention`, or `attn`.
            if hasattr(submodule, 'attn') or hasattr(submodule, 'self_attn'):
                # Print for debugging
                print(f"Replacing attention layer: {name} in {layer.__class__.__name__}")

                # Ensure the attention submodule has a `self` attention
                if hasattr(submodule, 'self'):
                    flash_attention = FlashSelfAttention()  # FlashAttention 2 replacement
                    setattr(submodule, 'self', flash_attention)  # Replace the self-attention
                elif hasattr(submodule, 'attn'):
                    flash_attention = FlashSelfAttention()
                    setattr(submodule, 'attn', flash_attention)

            
            # Recursively search and replace in nested modules
            replace_attention_with_flash(submodule)

    # Loop over all transformer layers in the model
    for name, module in model.encoder.layers():
        # Huggingface models often have transformer layers based on `BertLayer` or `TransformerEncoderLayer`.
        if isinstance(module, nn.Module):
            if hasattr(module, 'attention') or hasattr(module, 'self_attn'):
                print(f"Converting attention in module: {name}")
                replace_attention_with_flash(module)

    return model
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
    def freeze_blocks(self):
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
    
class FlashAttentionWrapper(nn.Module):
    def __init__(self, attention_layer):
        super(FlashAttentionWrapper, self).__init__()
        self.attention_layer = attention_layer
        
        # WavLM-specific attributes
        self.num_heads = attention_layer.num_heads
        self.head_dim = attention_layer.head_dim
        self.dropout = getattr(attention_layer, 'dropout', 0.1)  # default to 0.1 if not found
        
        # Preserve original projections
        self.query = attention_layer.q_proj
        self.key = attention_layer.k_proj
        self.value = attention_layer.v_proj
        self.out_proj = attention_layer.out_proj

        # Preserve position embeddings if they exist
        if hasattr(attention_layer, 'pos_bias_u'):
            self.pos_bias_u = attention_layer.pos_bias_u
            self.pos_bias_v = attention_layer.pos_bias_v
        
        if hasattr(attention_layer, 'pos_Q_proj'):
            self.pos_Q_proj = attention_layer.pos_Q_proj
        
        if hasattr(attention_layer, 'pos_K_proj'):
            self.pos_K_proj = attention_layer.pos_K_proj

        self.position_embedding_type = getattr(attention_layer, 'position_embedding_type', 'absolute')

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, past_key_value=None, use_cache=False, position_bias=False):
        batch_size, seq_length, _ = hidden_states.shape
        
        
        # Project input to q, k, v
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape
        query_layer = query_layer.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_layer = key_layer.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_layer = value_layer.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        

        # Handle position embeddings if necessary
        if self.position_embedding_type in ["relative_key", "relative_key_query"]:
            position_embeddings = self.compute_position_embeddings(seq_length)
            key_layer += position_embeddings
            if self.position_embedding_type == "relative_key_query":
                query_layer += position_embeddings

        try:
            if attention_mask is not None:
                # Expand mask to match the shape of the attention scores
                attention_mask = attention_mask[:, None, None, :].to(query_layer.dtype)
                # Apply the mask by adding a large negative value to the masked positions
                query_layer = query_layer + (attention_mask * -1e9)
            # Apply the attention mask within the flash attention function
            context_layer = flash_attn_func(query_layer, key_layer, value_layer, causal=False)
            
            # Reshape output
            context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
            
        except Exception as e:
            print(f"Error in flash_attn_func: {str(e)}")
            # Fallback to original attention mechanism
            return self.attention_layer(hidden_states, attention_mask, output_attentions=output_attentions, past_key_value=past_key_value, use_cache=use_cache)
        
        # Apply output projection
        attn_output = self.out_proj(context_layer)
        
        outputs = (hidden_states, attention_mask, output_attentions)


        return outputs 

    def compute_position_embeddings(self, seq_length):
        # Implement position embedding computation based on WavLM's approach
        # This is a placeholder and should be adjusted based on WavLM's specific implementation
        position_embeddings = torch.zeros(1, self.num_heads, seq_length, self.head_dim, device=self.query.weight.device)
        if hasattr(self, 'pos_bias_u') and hasattr(self, 'pos_bias_v'):
            position_embeddings += self.pos_bias_u[:, :, None, :] + self.pos_bias_v[:, :, :, None]
        return position_embeddings
class RespBertLSTMModel_flash(nn.Module):
    def __init__(self, bert_config, config):
        super(RespBertLSTMModel_flash, self).__init__()
        self.config = bert_config
        self.output = config['output_size']
        #self.wav_model = AutoModel.from_pretrained(config["model_name"])
        # Remove the last two encoder layers
        #self.wav_model  = load_wavlm_with_flash_attention2(config["model_name"])
        #print(self.wav_model)
        for i, layer in enumerate(self.wav_model.encoder.layers):
            if hasattr(layer, 'attention'):
                print(f"Replacing attention in layer {i}")
                layer.attention = FlashAttentionWrapper(layer.attention)


        self.d_model = bert_config.hidden_size
        self.features = config['hidden_units']
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.features, num_layers=config['n_lstm'], batch_first=True, dropout=0.2, bidirectional=True)
        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.features * 2, self.features * 2, kernel_size=5, dilation=2, padding="same"),
            nn.BatchNorm1d(self.features * 2),
            nn.GELU(),
            nn.Conv1d(self.features * 2, self.features * 2, kernel_size=3, padding="same"),
            nn.BatchNorm1d(self.features * 2),
            nn.GELU(),
            nn.Conv1d(self.features * 2, self.features, kernel_size=3, padding="same"),
            nn.BatchNorm1d(self.features),
            nn.GELU(),
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

        #input_values["input_values"] = input_values["input_values"]
        #input_values = input_values
        x = self.wav_model(**input_values)[0]
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

## MY PROPOSED MODEL DESIGNS
class RespBertLSTMModel(nn.Module):
    def __init__(self,bert_config,config):
        super(RespBertLSTMModel, self).__init__()
        self.config = bert_config
        self.output = config['output_size']
        

        self.wav_model = AutoModel.from_pretrained(config["model_name"])

        self.wav_model.encoder.layers = self.wav_model.encoder.layers[0:-2]

        self.d_model = bert_config.hidden_size
        
        self.features = config['hidden_units']
        
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.features, num_layers=config['n_lstm'], batch_first=True, dropout=0.2, bidirectional=True)
        
        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.features * 2 , self.features *2, kernel_size=5, dilation=2, padding="same"),
            nn.BatchNorm1d(self.features * 2),  
            nn.GELU(),
            

            nn.Conv1d(self.features * 2, self.features * 2, kernel_size=3, padding="same"),
            nn.BatchNorm1d(self.features * 2),  
            nn.GELU(),
            
            
            nn.Conv1d(self.features * 2, self.features, kernel_size=3, padding="same"),
            nn.BatchNorm1d(self.features),  
            nn.GELU(),            
            #nn.AdaptiveAvgPool1d(self.output),
  
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
    def __init__(self, bert_config, config):
        super(RespBertAttionModel, self).__init__()
        self.config = bert_config
        self.output = config['output_size']
        

        self.wav_model = AutoModel.from_pretrained(config["model_name"])

        self.wav_model.encoder.layers = self.wav_model.encoder.layers[0:-7]

        self.d_model = bert_config.hidden_size
        print(self.d_model)
        self.features = config['hidden_units']
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=16, dropout=0.1)
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
            nn.AdaptiveAvgPool1d(self.output),
  
        )

        self.feature_downsample = nn.Linear(self.features, 1)
        self.time = nn.Linear(self.output, self.output)

        #self.time = None
        self.tanh_va = nn.Tanh()
        self.flatten = nn.Flatten()      
        #self.init_weights()
        self.unfreeze_last_n_blocks(6)
                
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
    
#### TEST MODEL    
class RespBertLSTMModelTEST(nn.Module):
    def __init__(self, bert_config, config):
        super(RespBertLSTMModel, self).__init__()
        self.config = bert_config
        self.output = config['output_size']
        self.wav_model = AutoModel.from_pretrained(config["model_name"])
        self.wav_model.encoder.layers = self.wav_model.encoder.layers[0:-7]
        self.d_model = bert_config.hidden_size
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
        print(f"Input: {x.shape}")

        # Step 1: WavLM model
        # Input: (batch_size, audio_length)
        # Output: (batch_size, sequence_length, hidden_size) the first output [0] is the output from trensformerlayers and the second from the feature extraction of the cnn layers
        # the seqnce lenght is based on a step size of 20ms so if you have 30 seconds is 30/0.002 - 1500.
        # The WavLM model processes the raw audio input and returns contextualized representations
        x = self.wav_model(**input_values)
        print(f"After WavLM: {x.shape}")
        
        x = x[0]
        print(f"Getting the trensformer output form WavLM output: {x.shape}")


        # Step 2: LSTM layer
        # Input: (batch_size, sequence_length, hidden_size) 
        # Output: (batch_size, sequence_length, features)
        # The LSTM processes the sequence, maintaining the time dimension
        # so for each time step the lstm goes over the feature size of the model in this case 1024 and creates hidden layer 256 features. the next layer gets the information form the last layer hidden state
        x, _ = self.lstm(x)
        print(f"After LSTM: {x.shape}")

        # Step 3: Permute dimensions for Conv1d
        # Input: (batch_size, sequence_length, features)
        # Output: (batch_size, features, sequence_length)
        # Rearrange dimensions to apply 1D convolution over time
        x = x.permute(0, 2, 1)
        print(f"After permute: {x.shape}")

        # Step 4: Time downsample
        # Input: (batch_size, features, sequence_length)
        # Output: (batch_size, features, output_size)
        # Reduce the time dimension to a fixed size averga pooling to (output_size) and for finding patterns in the features from the lstm in the time demetion. 
        
        x = self.time_downsample(x)
        print(f"After time downsample: {x.shape}")

        # Step 5: Permute back
        # Input: (batch_size, features, output_size)
        # Output: (batch_size, output_size, features)
        # Rearrange dimensions for the linear layer
        x = x.permute(0, 2, 1)
        print(f"After permute back: {x.shape}")

        # Step 6: Feature downsample
        # Input: (batch_size, output_size, features)
        # Output: (batch_size, output_size, 1)
        # Reduce the feature dimension to 1 so for each timestep a linear layer is applied to have one demetion for each demention. afther this the demention should be (batch, 800, 1)
        x = self.feature_downsample(x)
        print(f"After feature downsample: {x.shape}")

        # Step 7: Flatten
        # Input: (batch_size, output_size, 1)
        # Output: (batch_size, output_size)
        # Remove the last dimension
        x = self.flatten(x)
        print(f"After flatten: {x.shape}")

        # Step 8: Final linear layer
        # Input: (batch_size, output_size)
        # Output: (batch_size, output_size)
        # Apply final transformation this to give the model the option to look in the time demention and see which one are inmportnat for the prediction.
        x = self.time(x)
        print(f"After final linear: {x.shape}")

        # Step 9: Tanh activation
        # Input: (batch_size, output_size)
        # Output: (batch_size, output_size)
        # Apply tanh activation to constrain values between -1 and 1
        x = self.tanh_va(x)
        print(f"Final output: {x.shape}")

        return x