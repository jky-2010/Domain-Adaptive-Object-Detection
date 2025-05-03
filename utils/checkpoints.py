"""
Author: Elias Mapendo
Date: May 03, 2025
Description:
utils/checkpoints.py utility for save checkpoint / load checkpoint.
"""

import torch

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """
    Save model, optimizer, scheduler state at a given checkpoint.

    Args:
        model (nn.Module): Model to save.
        optimizer (Optimizer): Optimizer to save.
        scheduler (LRScheduler): Learning rate scheduler to save.
        epoch (int): Current training epoch.
        path (str): File path to save the checkpoint.
    """
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(model, optimizer=None, scheduler=None, path=None, map_location='cuda'):
    """
    Load model and optionally optimizer/scheduler state from checkpoint.

    Args:
        model (nn.Module): Model to load weights into.
        optimizer (Optimizer, optional): Optimizer to load state into.
        scheduler (LRScheduler, optional): Scheduler to load state into.
        path (str): File path to load checkpoint from.
        map_location (str): Device to map the checkpoint.

    Returns:
        int: Epoch to resume from (if available), else 0.
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model'])

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return checkpoint.get('epoch', 0)
