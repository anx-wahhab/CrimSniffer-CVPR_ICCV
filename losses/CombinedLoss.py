import torch
import torch.nn as nn
import pytorch_msssim

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ssim = pytorch_msssim.SSIM(data_range=1.0, size_average=True)
        self.alpha = alpha

    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        ssim_loss = 1 - self.ssim(output, target)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss