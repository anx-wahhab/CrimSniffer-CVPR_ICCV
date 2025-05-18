import torch
import torch.nn as nn
import torchvision.models as models
from .mse import MSE  

# Utility function to compute Gram matrix
def gram_matrix(features):
    (b, c, h, w) = features.size()
    features = features.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)

# Define the VGG-19 model for feature extraction
class VGG19Features(nn.Module):
    def __init__(self, layers=None, device='cpu'):
        super(VGG19Features, self).__init__()
        self.device = device
        self.vgg19 = models.vgg19(pretrained=True).features.to(self.device)
        self.layers = layers if layers else ['0', '5', '10', '19', '28']
        
    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the same device as the model
        features = []
        for name, layer in self.vgg19._modules.items():
            x = layer(x)
            if name in self.layers:
                features.append(x)
        return features

# Define the StyleLoss class that uses Gram matrix
class StyleLoss(nn.Module):
    def __init__(self, layers=None, weights=None, device='cpu'):
        super(StyleLoss, self).__init__()
        self.device = device
        self.vgg = VGG19Features(layers, device)
        self.weights = weights if weights else [1.0 / len(self.vgg.layers)] * len(self.vgg.layers)
        self.mse = MSE()  

        
    def forward(self, generated, ground_truth):
        generated = generated.to(self.device)  # Ensure inputs are on the same device as the model
        ground_truth = ground_truth.to(self.device)
        
        generated_features = self.vgg(generated)
        ground_truth_features = self.vgg(ground_truth)
        
        style_loss = 0
        for gf, gt, w in zip(generated_features, ground_truth_features, self.weights):
            gram_generated = gram_matrix(gf)
            gram_ground_truth = gram_matrix(gt)
            style_loss += w * self.mse.compute(gram_generated, gram_ground_truth)
                    
        return style_loss
    
