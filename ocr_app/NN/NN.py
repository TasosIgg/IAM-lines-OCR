## NN.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import init
import math

class CRNN(nn.Module):
    """
    Advanced CRNN for IAM handwriting recognition with:
    - ResNet-inspired CNN backbone with residual connections
    - Spatial attention mechanism
    - Bidirectional LSTM layers with dropout
    - Optimized for 128x1024 input images
    """
    
    def __init__(self, vocab_size=80, hidden_size=256, num_lstm_layers=2, 
                 dropout=0.2, use_attention=True, attention_heads=4):
        super(CRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.use_attention = use_attention
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        
        # === CNN BACKBONE ===
        self.conv_layers = nn.Sequential(
            # Initial conv
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # Block 1: 128x1024 -> 64x512
            self._make_residual_block(64, 64),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 64x512 -> 32x256  
            self._make_residual_block(64, 128),
            self._make_residual_block(128, 128),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 32x256 -> 16x256 (preserve width)
            self._make_residual_block(128, 256),
            self._make_residual_block(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Block 4: 16x256 -> 8x256
            self._make_residual_block(256, 512),
            self._make_residual_block(512, 512),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Gradual height reduction: 8x256 -> 1x256
            nn.Conv2d(512, 512, (3, 1), (2, 1), (1, 0)),  # 8->4
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, (3, 1), (2, 1), (1, 0)),  # 4->2
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, None)),  # 2->1, preserve width
        )
        
        # === SPATIAL ATTENTION ===
        if self.use_attention:
            self.multihead_attention = MultiHeadSpatialAttention(
                512, attention_heads
            )
        
        # === RNN LAYERS ===
        self.rnn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(num_lstm_layers):
            input_size = 512 if i == 0 else hidden_size
            
            # Add LSTM layer
            self.rnn_layers.append(
                nn.LSTM(
                    input_size, hidden_size, 
                    bidirectional=True, batch_first=True
                )
            )
            
            # Add corresponding linear projection and dropout
            self.dropout_layers.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_size * 2, hidden_size),
                'dropout': nn.Dropout(dropout)
            }))
        
        # === CLASSIFIER ===
        self.classifier = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        """Create a residual block with consistent GELU activation"""
        layers = []
        
        # Main path
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        main_path = nn.Sequential(*layers)
        
        # Shortcut connection
        shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        return ResidualBlock(main_path, shortcut)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        init.orthogonal_(param)
                    elif 'bias' in name:
                        init.constant_(param, 0)
    
    def forward(self, x):
        # CNN feature extraction
        conv_features = self.conv_layers(x)  
        
        # Apply spatial attention if enabled
        if self.use_attention:
            conv_features = self.multihead_attention(conv_features)
        
        batch_size, channels, height, width = conv_features.size()
        assert height == 1, f"Height should be 1 after CNN, got {height}"
        
        rnn_input = conv_features.squeeze(2).permute(0, 2, 1)  
        
        # Pass through RNN layers with individual dropout
        rnn_output = rnn_input
        for i in range(self.num_lstm_layers):
            lstm_out, _ = self.rnn_layers[i](rnn_output)
            
            # Apply linear projection and dropout
            rnn_output = self.dropout_layers[i]['linear'](lstm_out)
            rnn_output = self.dropout_layers[i]['dropout'](rnn_output)
        
        # Final classification
        output = self.classifier(rnn_output)  
        output = F.log_softmax(output, dim=2)
        
        return output  #


class ResidualBlock(nn.Module):
    """Residual connection block with consistent GELU activation"""
    def __init__(self, main_path, shortcut):
        super(ResidualBlock, self).__init__()
        self.main_path = main_path
        self.shortcut = shortcut
    
    def forward(self, x):
        return F.gelu(self.main_path(x) + self.shortcut(x))


class MultiHeadSpatialAttention(nn.Module):
    """Multi-head spatial attention for better feature focusing"""
    
    def __init__(self, channels, num_heads=4):
        super(MultiHeadSpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.query_conv = nn.Conv2d(channels, channels, 1)
        self.key_conv = nn.Conv2d(channels, channels, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.output_conv = nn.Conv2d(channels, channels, 1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate Q, K, V
        q = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        k = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        v = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        
        # Transpose for attention computation
        q = q.permute(0, 1, 3, 2)  
        k = k.permute(0, 1, 2, 3)  
        v = v.permute(0, 1, 3, 2)  
        
        # Compute attention weights
        attention_weights = torch.matmul(q, k) / math.sqrt(self.head_dim)
        attention_weights = self.softmax(attention_weights)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended_values = torch.matmul(attention_weights, v)  
        
        # Concatenate heads and reshape
        attended_values = attended_values.permute(0, 1, 3, 2).contiguous()
        attended_values = attended_values.view(batch_size, channels, height, width)
        
        # Final projection
        output = self.output_conv(attended_values)
        
        # Residual connection
        return output + x

