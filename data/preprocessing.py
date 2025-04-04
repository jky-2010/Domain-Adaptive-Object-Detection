"""
Author: Elias Mapendo
Date: March 30, 2025
Description: Basic transformation class for Cityscapes dataset preprocessing.
This module provides standard transformations needed for preparing image and
annotation pairs for semantic segmentation training.
"""

import torchvision.transforms.functional as TF

class BasicTransform:
    def __init__(self, resize=(512, 1024)):
        self.resize = resize

    def __call__(self, image, annotation=None):
        # Resize the image
        image = TF.resize(image, self.resize)
        image = TF.to_tensor(image)

        # If annotation is given, process it
        if annotation is not None:
            annotation = TF.resize(annotation, self.resize, interpolation=TF.InterpolationMode.NEAREST)
            annotation = TF.pil_to_tensor(annotation).squeeze(0).long()
            return image, annotation

        return image
