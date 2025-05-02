"""
Author: Elias Mapendo
Date: April 22, 2025
Description:
Defines domain classifiers for domain adaptation in object detection.
Includes image-level and instance-level domain classifiers with integrated Gradient Reversal Layers (GRL).
"""

import torch.nn as nn
from models.advgrl import GradientReversalLayer


class ImageLevelDomainClassifier(nn.Module):
    """
    Classifies entire feature maps (image-level features) into source or target domains.
    Used for global feature alignment between domains.
    """

    def __init__(self, in_channels, hidden_dim=512):
        """
        Args:
            in_channels (int): Number of input feature channels (from backbone).
            hidden_dim (int): Size of hidden fully connected layers.
        """
        super(ImageLevelDomainClassifier, self).__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)

        # Fully connected layers after global average pooling
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Binary classification: source or target
        )

    def forward(self, x):
        # Global Average Pooling across spatial dimensions (H, W)
        x = x.mean(dim=[2, 3])  # (batch_size, channels)

        x = self.grl(x) # Apply Gradient Reversal Layer

        return self.classifier(x) # Classify as source/target


class InstanceLevelDomainClassifier(nn.Module):
    """
    Classifies RoI (Region of Interest) pooled features (instance-level features) into source or target.
    Used for object-level feature alignment.
    """

    def __init__(self, in_channels, hidden_dim=256):
        """
        Args:
            in_channels (int): Number of input channels after RoI Align.
            hidden_dim (int): Size of hidden layers.
        """
        super(InstanceLevelDomainClassifier, self).__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)

        # Fully connected network to classify instance features
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Source or Target
        )

    def forward(self, x):
        # x shape: (batch_size * num_rois, channels)

        # Apply Gradient Reversal Layer
        x = self.grl(x)

        return self.classifier(x) # Classify as source/target
