"""
Author: Elias Mapendo
Date: May 03, 2025
Description:
data/utils.py utility for collate function and dataloader construction.
"""

from torch.utils.data import DataLoader, Subset
import random, torch
from data.datasets import CityscapesDataset
from data.preprocessing import BasicTransform
from collections import OrderedDict

def collate_fn(batch):
    # Custom collate function for object detection datasets
    return tuple(zip(*batch))


def get_dataloaders(batch_size, target_labels, num_workers=2):
    """
    Prepare DataLoaders for source (clear) and target (foggy) domains.

    Args:
        batch_size (int): Number of samples per batch.
        target_labels (list): List of label IDs to include.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: source_loader, target_loader
    """
    transform = BasicTransform()

    # Clear weather (source domain)
    source_dataset = CityscapesDataset(mode='train', foggy=False, transforms=transform,
                                       target_labels=target_labels)
    subset_size_src = int(0.8 * len(source_dataset))
    source_subset = Subset(source_dataset, random.sample(range(len(source_dataset)), subset_size_src))

    # Foggy weather (target domain)
    target_dataset = CityscapesDataset(mode='train', foggy=True, transforms=transform,
                                       target_labels=target_labels)
    subset_size_tgt = int(0.8 * len(target_dataset))
    target_subset = Subset(target_dataset, random.sample(range(len(target_dataset)), subset_size_tgt))

    # Create DataLoaders
    source_loader = DataLoader(source_subset, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=num_workers)
    target_loader = DataLoader(target_subset, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=num_workers)

    return source_loader, target_loader


def ensure_ordered_dict(feats):
    """
    Converts various types of feature outputs into a consistent OrderedDict format.

    Args:
        feats: Can be a dict, list of feature maps, or a single tensor.

    Returns:
        OrderedDict: Feature maps with string keys ('0', '1', ..., 'pool').

    Raises:
        TypeError: If input type is unsupported.
    """
    if isinstance(feats, dict):
        # Already a dict — ensure keys are strings
        return OrderedDict((str(k), v) for k, v in feats.items())
    elif isinstance(feats, list):
        # Convert list to OrderedDict with expected FPN keys
        expected_keys_list = ['0', '1', '2', '3', 'pool']
        return OrderedDict((k, f) for k, f in zip(expected_keys_list, feats))
    elif isinstance(feats, torch.Tensor):
        # Single tensor case — wrap as dict with a default key
        return OrderedDict({'0': feats})
    else:
        raise TypeError(f"Unsupported feature type: {type(feats)}")
