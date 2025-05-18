import torch
import torch.nn as nn
import torchvision.utils as vutils
import random

class FeedbackLoopModule(nn.Module):
    def __init__(self, max_iterations=3, prob=0.5):
        super().__init__()
        self.components = nn.ModuleDict({
            'left_eye': LeftEyeComponent(),
            'right_eye': RightEyeComponent(),
            'nose': NoseComponent(),
            'mouth': MouthComponent(),
            'background': BackgroundComponent()
        })
        self.max_iterations = max_iterations
        self.prob = prob

    def forward(self, x):
        patches = self.crop(x)
        perturbed_images = self.create_perturbed_images(patches, x)
        return perturbed_images

    def crop(self, image):
        return {
            'left_eye': self.components['left_eye'].crop(image),
            'right_eye': self.components['right_eye'].crop(image),
            'nose': self.components['nose'].crop(image),
            'mouth': self.components['mouth'].crop(image),
            'background': self.components['background'].crop(image),
        }

    def create_perturbed_images(self, patches, original_image):
        perturbed_images = []

        # Create perturbed images for each component
        for component in patches.keys():
            perturbed_patches = patches.copy()
            perturbed_patches[component] = self.perturb(patches[component].clone())
            perturbed_image = self.decrop(perturbed_patches)
            perturbed_images.append(perturbed_image)

        # Create a perturbed image for the whole original image
        perturbed_full_image = self.perturb(original_image.clone())
        perturbed_images.append(perturbed_full_image)

        return perturbed_images

    def perturb(self, image):
        num_iterations = random.randint(1, self.max_iterations)
        refined_image = self.refine(image)
        for _ in range(num_iterations - 1):
            refined_image = self.refine(refined_image)
        return refined_image

    def refine(self, image):
        # Randomly apply perturbation with probability self.prob
        if random.random() < self.prob:
            perturbation_magnitude = random.uniform(0.05, 0.3)
            return image + torch.randn_like(image) * perturbation_magnitude
        else:
            return image

    def decrop(self, patches):
        blank_canvas = torch.zeros((patches['background'].shape[0], 3, 512, 512), device=patches['background'].device)
        blank_canvas[:, :, 0:512, 0:512] = patches['background']
        blank_canvas[:, :, 232:232+160, 182:182+160] = patches['nose']
        blank_canvas[:, :, 126:126+128, 108:108+128] = patches['left_eye']
        blank_canvas[:, :, 126:126+128, 255:255+128] = patches['right_eye']
        blank_canvas[:, :, 301:301+192, 169:169+192] = patches['mouth']
        return blank_canvas


class LeftEyeComponent(nn.Module):
    def __init__(self):
        super().__init__()
        self.crop_dimension = (108, 126, 236, 254)  # xmin, ymin, xmax, ymax

    def crop(self, image):
        assert image.shape[1:] == (3, 512, 512), f'[FeedbackLoop : LeftEye] Expected input shape {(-1, 3, 512, 512)}, but received {image.shape}.'
        return image[:, :, self.crop_dimension[1]:self.crop_dimension[3], self.crop_dimension[0]:self.crop_dimension[2]].clone()


class RightEyeComponent(nn.Module):
    def __init__(self):
        super().__init__()
        self.crop_dimension = (255, 126, 383, 254)  # xmin, ymin, xmax, ymax

    def crop(self, image):
        assert image.shape[1:] == (3, 512, 512), f'[FeedbackLoop : RightEye] Expected input shape {(-1, 3, 512, 512)}, but received {image.shape}.'
        return image[:, :, self.crop_dimension[1]:self.crop_dimension[3], self.crop_dimension[0]:self.crop_dimension[2]].clone()


class NoseComponent(nn.Module):
    def __init__(self):
        super().__init__()
        self.crop_dimension = (182, 232, 342, 392)  # xmin, ymin, xmax, ymax

    def crop(self, image):
        assert image.shape[1:] == (3, 512, 512), f'[FeedbackLoop : Nose] Expected input shape {(-1, 3, 512, 512)}, but received {image.shape}.'
        return image[:, :, self.crop_dimension[1]:self.crop_dimension[3], self.crop_dimension[0]:self.crop_dimension[2]].clone()


class MouthComponent(nn.Module):
    def __init__(self):
        super().__init__()
        self.crop_dimension = (169, 301, 361, 493)  # xmin, ymin, xmax, ymax

    def crop(self, image):
        assert image.shape[1:] == (3, 512, 512), f'[FeedbackLoop : Mouth] Expected input shape {(-1, 3, 512, 512)}, but received {image.shape}.'
        return image[:, :, self.crop_dimension[1]:self.crop_dimension[3], self.crop_dimension[0]:self.crop_dimension[2]].clone()


class BackgroundComponent(nn.Module):
    def __init__(self):
        super().__init__()
        self.crop_dimension = (0, 0, 512, 512)  # xmin, ymin, xmax, ymax

    def crop(self, image):
        assert image.shape[1:] == (3, 512, 512), f'[FeedbackLoop : Background] Expected input shape {(-1, 3, 512, 512)}, but received {image.shape}.'
        return image[:, :, self.crop_dimension[1]:self.crop_dimension[3], self.crop_dimension[0]:self.crop_dimension[2]].clone()

