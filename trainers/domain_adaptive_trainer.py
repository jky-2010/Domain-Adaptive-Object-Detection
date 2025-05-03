"""
Author: Elias Mapendo
Date: April 23, 2025
Description:
Domain-adaptive trainer for object detection using adversarial learning.
Combines Faster R-CNN detection loss with domain confusion losses from image-level and instance-level classifiers.
"""

import torch, os, sys, random
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import warnings

from models.faster_cnn import get_faster_rcnn_model
from models.domain_classifiers import ImageLevelDomainClassifier, InstanceLevelDomainClassifier
from models.domain_losses import compute_domain_loss
from data.datasets import CityscapesDataset
from data.preprocessing import BasicTransform
from data.utils import collate_fn
from data.preprocessing import ensure_three_channels
from utils.checkpoints import save_checkpoint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

        self.target_labels = [24, 25, 26, 27, 28, 31, 32, 33]
        self.num_classes = len(self.target_labels) + 1

        self.detector = get_faster_rcnn_model(num_classes=self.num_classes).to(self.device)
        pretrained_path = "experiments/faster_rcnn_cityscapes.pth"
        if os.path.exists(pretrained_path):
            print(f"[INFO] Loading base model weights from '{pretrained_path}'")
            state_dict = torch.load(pretrained_path, map_location=self.device)
            self.detector.load_state_dict(state_dict)
        else:
            print(f"[WARNING] Base model weights not found at '{pretrained_path}' — starting from scratch.")

        self.image_domain_classifier = ImageLevelDomainClassifier(in_channels=256).to(self.device)
        self.instance_domain_classifier = InstanceLevelDomainClassifier(in_channels=1024).to(self.device)

        params = list(self.detector.parameters()) + \
                 list(self.image_domain_classifier.parameters()) + \
                 list(self.instance_domain_classifier.parameters())

        self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

    def get_dataloaders(self):
        transform = BasicTransform()
        source_dataset = CityscapesDataset(mode='train', foggy=False, transforms=transform, target_labels=self.target_labels)
        subset_size_src = int(0.8 * len(source_dataset))
        source_subset = Subset(source_dataset, random.sample(range(len(source_dataset)), subset_size_src))

        target_dataset = CityscapesDataset(mode='train', foggy=True, transforms=transform, target_labels=self.target_labels)
        subset_size_tgt = int(0.8 * len(target_dataset))
        target_subset = Subset(target_dataset, random.sample(range(len(target_dataset)), subset_size_tgt))

        source_loader = DataLoader(source_subset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
        target_loader = DataLoader(target_subset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

        return source_loader, target_loader

    def train_one_epoch(self, source_loader, target_loader):
        self.detector.train()
        self.image_domain_classifier.train()
        self.instance_domain_classifier.train()

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        num_batches = min(len(source_loader), len(target_loader))
        total_loss = 0.0

        progress_bar = tqdm(range(num_batches), desc="Training Progress")

        for batch_idx in progress_bar:
            try:
                source_images, source_targets = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_images, source_targets = next(source_iter)

            source_images = [ensure_three_channels(img.to(self.device)) for img in source_images]
            source_targets = [{k: v.to(self.device) for k, v in t.items()} for t in source_targets]

            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            target_images = [ensure_three_channels(img.to(self.device)) for img in (target_batch[0] if isinstance(target_batch, tuple) else target_batch)]

            self.optimizer.zero_grad()
            loss_dict = self.detector(source_images, source_targets)
            detection_loss = sum(loss for loss in loss_dict.values())

            source_tensor = torch.stack(source_images)
            target_tensor = torch.stack(target_images)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                source_features = self.detector.backbone(source_tensor)
                target_features = self.detector.backbone(target_tensor)

            src_img_features = source_features['0'] if isinstance(source_features, dict) else source_features[0]
            tgt_img_features = target_features['0'] if isinstance(target_features, dict) else target_features[0]

            src_img_preds = self.image_domain_classifier(src_img_features)
            tgt_img_preds = self.image_domain_classifier(tgt_img_features)

            img_domain_loss = compute_domain_loss(src_img_preds, tgt_img_preds, self.device)

            if not isinstance(source_features, dict):
                source_feats_dict = {str(i): f for i, f in enumerate(source_features)}
                target_feats_dict = {str(i): f for i, f in enumerate(target_features)}
            else:
                source_feats_dict = source_features
                target_feats_dict = target_features

            try:
                with torch.no_grad():
                    src_box_features = self.detector.roi_heads.box_roi_pool(
                        source_feats_dict,
                        [self.detector.rpn.box_coder.decode(p.detach()) for p in self.detector.rpn.head.anchors],
                        [source_tensor.shape[-2:]] * len(source_tensor)
                    )
                    tgt_box_features = self.detector.roi_heads.box_roi_pool(
                        target_feats_dict,
                        [self.detector.rpn.box_coder.decode(p.detach()) for p in self.detector.rpn.head.anchors],
                        [target_tensor.shape[-2:]] * len(target_tensor)
                    )
                    src_proposal_feats = self.detector.roi_heads.box_head(src_box_features)
                    tgt_proposal_feats = self.detector.roi_heads.box_head(tgt_box_features)
            except Exception as e:
                print(f"[WARNING] Error in instance-level feature extraction: {e}")
                print("[INFO] Skipping instance-level domain adaptation for this batch")
                instance_domain_loss = torch.tensor(0.0, device=self.device)
                src_proposal_feats, tgt_proposal_feats = [], []

            instance_domain_loss_src = 0.0
            instance_domain_loss_tgt = 0.0

            if hasattr(src_proposal_feats, 'size') and src_proposal_feats.size(0) > 0:
                src_inst_preds = self.instance_domain_classifier(src_proposal_feats)
                src_inst_labels = torch.zeros(src_inst_preds.shape[0], dtype=torch.long, device=self.device)
                instance_domain_loss_src = torch.nn.functional.cross_entropy(src_inst_preds, src_inst_labels)

            if hasattr(tgt_proposal_feats, 'size') and tgt_proposal_feats.size(0) > 0:
                tgt_inst_preds = self.instance_domain_classifier(tgt_proposal_feats)
                tgt_inst_labels = torch.ones(tgt_inst_preds.shape[0], dtype=torch.long, device=self.device)
                instance_domain_loss_tgt = torch.nn.functional.cross_entropy(tgt_inst_preds, tgt_inst_labels)

            instance_domain_loss = (instance_domain_loss_src + instance_domain_loss_tgt
                                     if isinstance(instance_domain_loss_src, torch.Tensor) and isinstance(instance_domain_loss_tgt, torch.Tensor)
                                     else torch.tensor(0.0, device=self.device))

            total_batch_loss = detection_loss + img_domain_loss + instance_domain_loss
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()
            progress_bar.set_postfix({"Loss": f"{total_batch_loss.item():.4f}"})

        self.lr_scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"\n[INFO] Average Training Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self, num_epochs=25):
        source_loader, target_loader = self.get_dataloaders()
        try:
            for epoch in range(num_epochs):
                print(f"\n[INFO] Epoch {epoch+1}/{num_epochs}")
                self.train_one_epoch(source_loader, target_loader)
                if (epoch + 1) % 5 == 0:
                    os.makedirs('experiments/checkpoints', exist_ok=True)
                    save_checkpoint(self.detector, self.optimizer, self.lr_scheduler, epoch + 1,
                                    f'experiments/checkpoints/faster_rcnn_da_epoch{epoch + 1}.pth')
                    print(f"[INFO] Saved checkpoint at epoch {epoch+1}")
        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user. Saving final model...")
        except Exception as e:
            print(f"\n[ERROR] Training failed with exception: {e}")
            raise

        print("\n[INFO] Domain-Adaptive Training Complete!")
        return self.detector