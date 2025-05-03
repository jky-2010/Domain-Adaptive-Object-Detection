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
    Ensure the image tensor has exactly 3 channels
    Args:
        image_tensor (torch.Tensor): Input image tensor
    Returns:
        torch.Tensor: Image tensor with 3 channels
    """
    if image_tensor.dim() == 3:  # Single image
        if image_tensor.size(0) == 1:
            # Convert single channel to 3 channels
            return image_tensor.repeat(3, 1, 1)
        elif image_tensor.size(0) == 2:
            # Add a third channel
            return torch.cat([image_tensor, torch.zeros(1, image_tensor.size(1), image_tensor.size(2),
                                                      device=image_tensor.device)], dim=0)
        elif image_tensor.size(0) == 3:
            # Already 3 channels
            return image_tensor
        else:
            # Truncate to 3 channels if more
            return image_tensor[:3, :, :]
    elif image_tensor.dim() == 4:  # Batch of images
        if image_tensor.size(1) == 1:
            # Convert single channel to 3 channels
            return image_tensor.repeat(1, 3, 1, 1)
        elif image_tensor.size(1) == 2:
            # Add a third channel
            zeros = torch.zeros(image_tensor.size(0), 1, image_tensor.size(2), image_tensor.size(3),
                               device=image_tensor.device)
            return torch.cat([image_tensor, zeros], dim=1)
        elif image_tensor.size(1) == 3:
            # Already 3 channels
            return image_tensor
        else:
            # Truncate to 3 channels if more
            return image_tensor[:, :3, :, :]
    return image_tensor