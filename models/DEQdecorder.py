import torch
import torch.nn as nn
from . import block

class DEQLayer(nn.Module):
    def __init__(self, func, max_iter=50, tol=1e-4):
        super(DEQLayer, self).__init__()
        self.func = func
        self.max_iter = max_iter
        self.tol = tol

    def forward(self, x):
        z = torch.zeros_like(x)
        z = self.broyden_solver(self.func, z, x)
        return z

    def broyden_solver(self, func, z, x):
        for _ in range(self.max_iter):
            z_new = func(z, x)
            if torch.norm(z_new - z) < self.tol:
                break
            z = z_new
        return z

class ReflectionPad2d(nn.Module):
    def __init__(self, source_dimension, target_dimension):
        super(ReflectionPad2d, self).__init__()
        dif = target_dimension - source_dimension
        padding_left = padding_right = padding_top = padding_bottom = dif // 2
        if dif % 2: 
            padding_right = padding_bottom = (dif // 2) + 1
        self.pad = nn.ReflectionPad2d((padding_left, padding_right, padding_top, padding_bottom))

    def forward(self, x):
        return self.pad(x)

class MultiScaleDecoder(nn.Module):
    def __init__(self, channels, input_dimension, output_dimension, latent_dimension):
        super(MultiScaleDecoder, self).__init__()
        self.channels = channels
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.conv_dimension = input_dimension
        self.latent_dimension = latent_dimension
        
        self.decoder_list = nn.ModuleList()
        for i in range(1, len(self.channels) - 1):
            self.decoder_list.append(block.ResNet(self.channels[i-1]))
            self.decoder_list.append(block.ConvTrans2D(self.channels[i-1], self.channels[i]))
            self.conv_dimension = self.conv_dimension * 2

        self.latent = nn.Linear(self.latent_dimension, self.channels[0] * self.input_dimension * self.input_dimension)
        
        self.deq = DEQLayer(self.deq_func)
        
        self.output = nn.Sequential(
            block.ResNet(self.channels[-2]),
            ReflectionPad2d(self.conv_dimension, self.output_dimension),
            nn.Conv2d(self.channels[-2], self.channels[-1], kernel_size=4, stride=1, padding='same', padding_mode='reflect')
        )

    def deq_func(self, z, x):
        for decoder in self.decoder_list:
            x = decoder(x)
        
        # Ensure x and z have the same spatial dimensions
        if x.size() != z.size():
            x = torch.nn.functional.interpolate(x, size=z.size()[2:])
        
        # Now perform addition after ensuring dimensions match
        try:
            return x + z
        except RuntimeError as e:
            print(f"Error: {e}")

    def forward(self, x):
        x = self.latent(x)
        x = torch.reshape(x, (x.shape[0], self.channels[0], self.input_dimension, self.input_dimension))
        x = self.deq(x)
        x = self.output(x)
        return x
