"""
Author: Elias Mapendo
Date: April 23, 2025
Description:
Domain-adaptive trainer for object detection using adversarial learning.
Combines Faster R-CNN detection loss with domain confusion losses from image-level and instance-level classifiers.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.faster_cnn import get_faster_rcnn_model
from models.domain_classifiers import ImageLevelDomainClassifier, InstanceLevelDomainClassifier
from data.datasets import CityscapesDataset
from data.preprocessing import BasicTransform

def collate_fn(batch):
    return tuple(zip(*batch))

class DomainAdaptiveTrainer:
    """
    Trainer class that handles domain-adaptive training
    by combining detection loss + domain classification loss
    for both clear and foggy datasets.
    """
    def __init__(self, device='cuda', batch_size=2, lr=0.005):
        self.device = device
        self.batch_size = batch_size
        self.lr = lr

        # Label mapping
        self.target_labels = [24, 25, 26, 27, 28, 31, 32, 33]
        self.num_classes = len(self.target_labels) + 1  # +1 for background

        # Load base Faster R-CNN model
        self.detector = get_faster_rcnn_model(num_classes=self.num_classes).to(self.device)

        # Add domain classifiers
        self.image_domain_classifier = ImageLevelDomainClassifier(in_channels=256).to(self.device)
        self.instance_domain_classifier = InstanceLevelDomainClassifier(in_channels=1024).to(self.device)

        # Combine all parameters for joint optimization
        params = list(self.detector.parameters()) + \
                 list(self.image_domain_classifier.parameters()) + \
                 list(self.instance_domain_classifier.parameters())

        self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

        # Loss functions
        self.domain_loss_fn = torch.nn.CrossEntropyLoss()

    def get_dataloaders(self):
        """Prepare DataLoaders for source (clear) and target (foggy) domains."""
        transform = BasicTransform()

        source_dataset = CityscapesDataset(mode='train', foggy=False, transforms=transform, target_labels=self.target_labels)
        target_dataset = CityscapesDataset(mode='train', foggy=True, transforms=transform, target_labels=self.target_labels)

        source_loader = DataLoader(source_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
        target_loader = DataLoader(target_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

        return source_loader, target_loader

    def train_one_epoch(self, source_loader, target_loader):
        """Train for one epoch, alternating between source and target batches."""
        self.detector.train()
        self.image_domain_classifier.train()
        self.instance_domain_classifier.train()

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        num_batches = min(len(source_loader), len(target_loader))
        total_loss = 0.0

        progress_bar = tqdm(range(num_batches), desc="Training Progress")

        for _ in progress_bar:
            # Get source batch (with annotations)
            try:
                source_images, source_targets = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_images, source_targets = next(source_iter)

            source_images = [img.to(self.device) for img in source_images]
            source_targets = [{k: v.to(self.device) for k, v in t.items()} for t in source_targets]

            # Get target batch (no annotations)
            try:
                target_images = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_images = next(target_iter)

            target_images = [img.to(self.device) for img in target_images]

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass: source
            source_features = self.detector.backbone(torch.stack(source_images))
            source_rpn_out, _ = self.detector.rpn(source_features, None)

            # Forward pass: target
            target_features = self.detector.backbone(torch.stack(target_images))
            target_rpn_out, _ = self.detector.rpn(target_features, None)

            # Compute Detection Loss (only for source domain)
            loss_dict = self.detector(source_images, source_targets)
            detection_loss = sum(loss for loss in loss_dict.values())

            # Compute Image-Level Domain Loss
            src_img_preds = self.image_domain_classifier(source_features['0'])
            tgt_img_preds = self.image_domain_classifier(target_features['0'])

            src_img_labels = torch.zeros(src_img_preds.shape[0], dtype=torch.long, device=self.device)  # 0 = source
            tgt_img_labels = torch.ones(tgt_img_preds.shape[0], dtype=torch.long, device=self.device)   # 1 = target

            img_domain_loss = self.domain_loss_fn(src_img_preds, src_img_labels) + \
                              self.domain_loss_fn(tgt_img_preds, tgt_img_labels)

            # Compute Instance-Level Domain Loss
            # Note: Faster R-CNN returns proposals
            src_instance_features = source_rpn_out['proposals'][0]
            tgt_instance_features = target_rpn_out['proposals'][0]

            if src_instance_features.size(0) > 0:
                src_inst_preds = self.instance_domain_classifier(src_instance_features)
                src_inst_labels = torch.zeros(src_inst_preds.shape[0], dtype=torch.long, device=self.device)
                instance_domain_loss_src = self.domain_loss_fn(src_inst_preds, src_inst_labels)
            else:
                instance_domain_loss_src = 0.0

            if tgt_instance_features.size(0) > 0:
                tgt_inst_preds = self.instance_domain_classifier(tgt_instance_features)
                tgt_inst_labels = torch.ones(tgt_inst_preds.shape[0], dtype=torch.long, device=self.device)
                instance_domain_loss_tgt = self.domain_loss_fn(tgt_inst_preds, tgt_inst_labels)
            else:
                instance_domain_loss_tgt = 0.0

            instance_domain_loss = instance_domain_loss_src + instance_domain_loss_tgt

            # Total Loss
            total_batch_loss = detection_loss + img_domain_loss + instance_domain_loss

            # Backward and step
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()

        self.lr_scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"\n[INFO] Average Training Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self, num_epochs=10):
        """Train the model across multiple epochs."""
        source_loader, target_loader = self.get_dataloaders()

        for epoch in range(num_epochs):
            print(f"\n[INFO] Epoch {epoch+1}/{num_epochs}")
            self.train_one_epoch(source_loader, target_loader)

        print("\n[INFO] Domain-Adaptive Training Complete!")
        return self.detector  # Return adapted detector