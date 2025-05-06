"""
Author: Elias Mapendo
Date: May 03, 2025
Description:
data/utils.py utility for collate fn.
"""

from torch.utils.data import DataLoader, Subset
import random
from data.datasets import CityscapesDataset
from data.preprocessing import BasicTransform

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloaders(self):
    """Prepare DataLoaders for source (clear) and target (foggy) domains."""
    transform = BasicTransform()

    # SOURCE (clear)
    source_dataset = CityscapesDataset(mode='train', foggy=False, transforms=transform,
                                       target_labels=self.target_labels)
    subset_size_src = int(0.8 * len(source_dataset))
    source_subset = Subset(source_dataset, random.sample(range(len(source_dataset)), subset_size_src))

    # TARGET (foggy)
    target_dataset = CityscapesDataset(mode='train', foggy=True, transforms=transform,
                                       target_labels=self.target_labels)
    subset_size_tgt = int(0.8 * len(target_dataset))
    target_subset = Subset(target_dataset, random.sample(range(len(target_dataset)), subset_size_tgt))

    source_loader = DataLoader(source_subset, batch_size=self.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=2)
    target_loader = DataLoader(target_subset, batch_size=self.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=2)

    return source_loader, target_loader