import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from .mse import MSE  

class SelfSupervisedModule(nn.Module):
    def __init__(self, data_range=1.0):
        super(SelfSupervisedModule, self).__init__()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range)
        self.mse = MSE()  

    def forward(self, input_sketch, generated_image):
        self_supervised_loss = self.calculate_loss(input_sketch, generated_image)
        return self_supervised_loss

    def calculate_ssim(self, img1, img2):
        # Reset and compute SSIM
        self.ssim_metric.reset()
        self.ssim_metric.update(img1, img2)
        ssim_score = self.ssim_metric.compute()
        return 1 - ssim_score  # Subtract SSIM from 1 to use as a loss

    def calculate_loss(self, input_sketch, generated_image):
        # Convert images to grayscale by averaging across the color channels
        input_sketch_gray = torch.mean(input_sketch, dim=1, keepdim=True)
        generated_image_gray = torch.mean(generated_image, dim=1, keepdim=True)

        mse_loss = self.mse.compute(generated_image_gray, input_sketch_gray)
        ssim_loss = self.calculate_ssim(input_sketch_gray, generated_image_gray)

        # Total loss is the sum of MSE loss and SSIM loss
        total_loss = mse_loss + ssim_loss
        return total_loss
