"""
Author: Elias Mapendo
Date: March 30, 2025
Description: Cityscapes dataset loader for semantic segmentation tasks.
This module provides a PyTorch Dataset implementation for the Cityscapes dataset,
supporting both training with annotations and inference without annotations.
"""

import os, torch
from torch.utils.data import Dataset
from PIL import Image
from utils.segmentation_to_bbox import masks_to_boxes

class CityscapesDataset(Dataset):
    def __init__(self, mode='train', foggy=False, transforms=None, target_labels=None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.transforms = transforms
        self.mode = mode
        self.target_labels = target_labels if target_labels else [24, 25, 26, 27, 28, 31, 32, 33]  # Common object classes

        if foggy:
            colab_path = '/content/gdrive/MyDrive/leftImg8bit_foggy'
            local_path = '/Users/eliasmapendo/Google Drive/My Drive/leftImg8bit_foggy/'

            base_img_dir = local_path if os.path.exists(local_path) else colab_path
            img_dir = os.path.join(base_img_dir, mode)
            self.annotations_available = False
        else:
            img_dir = os.path.join(project_root, 'data', 'cityscapes', 'leftImg8bit', mode)
            ann_dir = os.path.join(project_root, 'data', 'cityscapes', 'gtFine', mode)
            self.annotations_available = True
            self.annotations = []

        self.images = []

        for city in sorted(os.listdir(img_dir)):
            city_dir = os.path.join(img_dir, city)
            if os.path.isdir(city_dir):
                for file_name in sorted(os.listdir(city_dir)):
                    if file_name.endswith('.png'):
                        self.images.append(os.path.join(city_dir, file_name))

        if self.annotations_available:
            for city in sorted(os.listdir(ann_dir)):
                city_ann_dir = os.path.join(ann_dir, city)
                if os.path.isdir(city_ann_dir):
                    for file_name in sorted(os.listdir(city_ann_dir)):
                        if file_name.endswith('_gtFine_labelIds.png'):
                            self.annotations.append(os.path.join(city_ann_dir, file_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')  # ✅ always RGB

        if self.annotations_available:
            annotation = Image.open(self.annotations[idx])
            if self.transforms:
                img, annotation = self.transforms(img, annotation)

            boxes, labels = masks_to_boxes(annotation, self.target_labels)

            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx])
            }
            return img, target

        else:
            # ✅ Always call transforms(img, annotation=None)
            if self.transforms:
                img = self.transforms(img, annotation=None)
            return img