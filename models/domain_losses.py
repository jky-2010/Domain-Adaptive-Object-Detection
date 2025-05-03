"""
Author: Elias Mapendo
Date: May 03, 2025
Description:
models/domain_losses.py utility for compute domain loss.
"""

import torch
import torch.nn as nn

def compute_domain_loss(src_preds, tgt_preds, device):
    """
    Computes the total domain classification loss for source and target predictions.

    Parameters:
        src_preds (Tensor): Predictions from the domain classifier for source images.
        tgt_preds (Tensor): Predictions from the domain classifier for target images.
        device (str): Device to place labels on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Combined domain classification loss.
    """
    domain_loss_fn = nn.CrossEntropyLoss()

    src_labels = torch.zeros(src_preds.shape[0], dtype=torch.long, device=device)
    tgt_labels = torch.ones(tgt_preds.shape[0], dtype=torch.long, device=device)

    return domain_loss_fn(src_preds, src_labels) + domain_loss_fn(tgt_preds, tgt_labels)

