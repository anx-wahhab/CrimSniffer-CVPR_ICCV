import torch
import torch.nn as nn
import torch.nn.functional as F
from . import block

class Encoder(nn.Module):
    def __init__(self, channels, input_dimension, latent_dimension, use_attention=True):
        super().__init__()

        self.channels = channels
        self.input_dimension = input_dimension
        self.conv_dimension = input_dimension
        self.latent_dimension = latent_dimension
        self.use_attention = use_attention
        
        self.encoder_list = nn.ModuleList()
        for i in range(1, len(self.channels)):
            self.encoder_list.append(block.Conv2D(self.channels[i-1], self.channels[i]))
            self.encoder_list.append(block.ResNet(self.channels[i]))
            self.conv_dimension = self.conv_dimension // 2

        self.attention = None
        if self.use_attention:
            self.attention = CBAM(self.channels[-1])

        self.latent = nn.Linear(self.channels[-1] * self.conv_dimension * self.conv_dimension, self.latent_dimension)

    def forward(self, x):
        for encoder in self.encoder_list:
            x = encoder(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        x = torch.flatten(x, 1)
        x = self.latent(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.global_avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.global_max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x_out = self.channel_attention(x)
        x_out = self.spatial_attention(x_out)
        return x_out

class Decoder(nn.Module):
    
    def ReflectionPad2d(self, source_dimension, target_dimension):
        dif = target_dimension - source_dimension
        padding_left = padding_right = padding_top = padding_bottom = dif // 2
        if dif%2: padding_right = padding_bottom = (dif // 2) + 1
        return nn.ReflectionPad2d((padding_left, padding_right, padding_top, padding_bottom))
    
    def __init__(self, channels, input_dimension, output_dimension, latent_dimension):
        super().__init__()
        
        self.channels = channels
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.conv_dimension = input_dimension
        self.latent_dimension = latent_dimension
        
        self.decoder_list = nn.ModuleList()
        for i in range(1, len(self.channels)-1):
            self.decoder_list.append(block.ResNet(self.channels[i-1]))
            self.decoder_list.append(block.ConvTrans2D(self.channels[i-1], self.channels[i]))
            self.conv_dimension = self.conv_dimension * 2
        
        self.latent = nn.Linear(self.latent_dimension, self.channels[0] * self.input_dimension * self.input_dimension)
        
        self.output = nn.Sequential(
            block.ResNet(self.channels[-2]),
            self.ReflectionPad2d(self.conv_dimension, self.output_dimension),
            nn.Conv2d(self.channels[-2], self.channels[-1], kernel_size=4, stride=1, padding='same', padding_mode='reflect')
        )
        
    def forward(self, x):
        x = self.latent(x)
        x = torch.reshape(x, (x.shape[0], self.channels[0], self.input_dimension, self.input_dimension))
        for decoder in self.decoder_list:
            x = decoder(x)
        x = self.output(x)
        return x

