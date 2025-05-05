"""
Author: Elias Mapendo
Date: March 30, 2025
Description: Basic transformation class for Cityscapes dataset preprocessing.
This module provides standard transformations needed for preparing image and
annotation pairs for semantic segmentation training.
"""

import torchvision.transforms.functional as TF
import torch

class BasicTransform:
    def __init__(self, resize=(768, 1536)):
        self.resize = resize

    def __call__(self, image, annotation=None):
        # Resize image and convert to tensor
        image = TF.resize(image, self.resize)
        image = TF.to_tensor(image)

        # Ensure all images are 3-channel
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 2:
            # Add a third channel with zeros for any 2-channel images
            image = torch.cat([image, torch.zeros(1, image.shape[1], image.shape[2])], dim=0)
        elif image.shape[0] > 3:
            # If more than 3 channels, truncate to 3
            image = image[:3, :, :]
        elif image.shape[0] != 3:
            raise ValueError(f"Unexpected channel count: {image.shape[0]}")

        if annotation is not None:
            annotation = TF.resize(annotation, self.resize, interpolation=TF.InterpolationMode.NEAREST)
            annotation = TF.pil_to_tensor(annotation).squeeze(0).long()
            return image, annotation

        return image


def ensure_three_channels(image_tensor):
    """
    Ensure the image tensor has exactly 3 channels.
    Args:
        image_tensor (torch.Tensor): Input image tensor.
    Returns:
        torch.Tensor: Image tensor with 3 channels.
    """
    if image_tensor.dim() == 3:  # Single image: [C, H, W]
        c, h, w = image_tensor.shape
        if c == 1:
            return image_tensor.repeat(3, 1, 1)
        elif c == 2:
            return torch.cat([image_tensor, torch.zeros(1, h, w, device=image_tensor.device)], dim=0)
        elif c >= 3:
            return image_tensor[:3, :, :]
    elif image_tensor.dim() == 4:  # Batch of images: [B, C, H, W]
        b, c, h, w = image_tensor.shape
        if c == 1:
            return image_tensor.repeat(1, 3, 1, 1)
        elif c == 2:
            zeros = torch.zeros(b, 1, h, w, device=image_tensor.device)
            return torch.cat([image_tensor, zeros], dim=1)
        elif c >= 3:
            return image_tensor[:, :3, :, :]

    raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape: {image_tensor.shape}")