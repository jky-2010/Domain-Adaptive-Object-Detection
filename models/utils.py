"""
Author: Elias Mapendo
Date: May 06, 2025
Description:
model/utils.py utility for ensuring feature maps are in OrderedDict and utility for compute domain loss.
"""

from collections import OrderedDict
import torch
import torch.nn as nn

def get_proposals_from_rpn(detector, features, image_list, targets=None):
    """
    Ensures RPN feature maps are in OrderedDict format and calls the detector's RPN module.

    Args:
        detector (nn.Module): The Faster R-CNN detector.
        features (dict | list | Tensor): Feature maps from the backbone.
        image_list (ImageList): Transformed image list object from the detector.
        targets (list[dict] | None): Optional targets for the RPN.

    Returns:
        proposals (list[Tensor]), rpn_losses (dict): Output from RPN forward call.
    """
    if not isinstance(features, OrderedDict):
        features = OrderedDict((str(k), v) for k, v in enumerate(features))
    return detector.rpn(image_list, features, targets=targets)


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