"""
Author: Elias Mapendo
Date: April 23, 2025
Description:
Domain-adaptive trainer for object detection using adversarial learning.
Combines Faster R-CNN detection loss with domain confusion losses from image-level and instance-level classifiers.
"""

import torch, os, sys, random, warnings
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import OrderedDict
from models.faster_cnn import get_faster_rcnn_model
from models.domain_classifiers import ImageLevelDomainClassifier, InstanceLevelDomainClassifier
from data.datasets import CityscapesDataset
from data.preprocessing import BasicTransform
from data.preprocessing import ensure_three_channels
from models.domain_losses import compute_domain_loss
from data.utils import collate_fn

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

        # Label mapping
        self.target_labels = [24, 25, 26, 27, 28, 31, 32, 33]
        self.num_classes = len(self.target_labels) + 1  # +1 for background

        # Save original RPN method, Load base Faster R-CNN model
        self.detector = get_faster_rcnn_model(num_classes=self.num_classes).to(self.device)

        # Load pretrained base model weights if available
        pretrained_path = "experiments/faster_rcnn_cityscapes.pth"
        if os.path.exists(pretrained_path):
            print(f"[INFO] Loading base model weights from '{pretrained_path}'")
            state_dict = torch.load(pretrained_path, map_location=self.device)
            self.detector.load_state_dict(state_dict)
        else:
            print(f"[WARNING] Base model weights not found at '{pretrained_path}' — starting from scratch.")

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

    def get_proposals_from_rpn(self, detector, images, features, targets=None):
        """
        Safely call RPN using the expected input types.
        """
        image_list, _ = detector.transform(images)

        if not isinstance(features, OrderedDict):
            features = OrderedDict((str(k), v) for k, v in enumerate(features))

        return detector.rpn(features, image_list, targets=targets)

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

        for batch_idx in progress_bar:
            # === Source (clear) ===
            try:
                source_images, source_targets = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_images, source_targets = next(source_iter)

            source_images = [img.to(self.device) for img in source_images]
            source_targets = [{k: v.to(self.device) for k, v in t.items()} for t in source_targets]

            # === Target (foggy, no annotations) ===
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            # Handle different return types from target_batch
            if isinstance(target_batch, tuple) and len(target_batch) > 0:
                target_images = target_batch[0]
            else:
                target_images = target_batch

            target_images = [img.to(self.device) for img in target_images]

            # Check and fix each image individually to ensure 3 channels
            for i in range(len(source_images)):
                source_images[i] = ensure_three_channels(source_images[i])

            for i in range(len(target_images)):
                if target_images[i].dim() == 2:
                    target_images[i] = target_images[i].unsqueeze(0)  # [H, W] -> [1, H, W]
                target_images[i] = ensure_three_channels(target_images[i])

            for i in range(len(source_images)):
                if source_images[i].dim() == 2:
                    source_images[i] = source_images[i].unsqueeze(0)
                source_images[i] = ensure_three_channels(source_images[i])

            # === Forward: Compute detection loss (source only) ===
            self.optimizer.zero_grad()

            # Use original source images for detection to maintain consistency
            # with the pretrained model
            loss_dict = self.detector(source_images, source_targets)
            detection_loss = sum(loss for loss in loss_dict.values())

            # === Extract features ===
            # Stack and ensure 3 channels for both source and target tensors
            source_tensor = torch.stack(source_images)
            target_tensor = torch.stack(target_images)

           # Print shape info for debugging (only on first batch)
            if batch_idx == 0:
                print(f"[INFO] Source tensor shape: {source_tensor.shape}")
                print(f"[INFO] Target tensor shape: {target_tensor.shape}")

            # Extract backbone features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                source_features = self.detector.backbone(source_tensor)
                target_features = self.detector.backbone(target_tensor)

            if batch_idx == 0:
                print(f"[DEBUG] source_features keys: {list(source_features.keys())}")
                print(f"[DEBUG] source_features type: {type(source_features)}")

            expected_keys = {'0', '1', '2', '3', 'pool'}
            assert set(source_features.keys()).issuperset(
                expected_keys), f"Backbone keys mismatch: {source_features.keys()}"

            # === Image-level domain loss ===
            # Get the first feature map from the backbone
            if isinstance(source_features, dict):
                # If features are a dictionary, get the first item
                src_img_features = list(source_features.values())[0]
                tgt_img_features = list(target_features.values())[0]
            else:
                # If features are a list, get the first item
                src_img_features = source_features[0]
                tgt_img_features = target_features[0]

            src_img_preds = self.image_domain_classifier(src_img_features)
            tgt_img_preds = self.image_domain_classifier(tgt_img_features)

            img_domain_loss = compute_domain_loss(src_img_preds, tgt_img_preds, self.device)

            try:
                with torch.no_grad():
                    # Prepare image lists using detector's internal transform
                    src_image_list, _ = self.detector.transform(source_images)
                    tgt_image_list, _ = self.detector.transform(target_images)

                    src_feats_raw = self.detector.backbone(src_image_list.tensors)
                    tgt_feats_raw = self.detector.backbone(tgt_image_list.tensors)

                    print(f"[DEBUG] Raw src_feats type: {type(src_feats_raw)}")
                    if isinstance(src_feats_raw, dict):
                        print(f"[DEBUG] Raw src_feats keys: {list(src_feats_raw.keys())}")
                    elif isinstance(src_feats_raw, list):
                        print(f"[DEBUG] Raw src_feats list length: {len(src_feats_raw)}")
                        print(f"[DEBUG] Raw src_feats[0] shape: {src_feats_raw[0].shape}")
                    elif isinstance(src_feats_raw, torch.Tensor):
                        print(f"[DEBUG] Raw src_feats shape: {src_feats_raw.shape}")
                    else:
                        print(f"[DEBUG] Raw src_feats unknown structure: {src_feats_raw}")

                    # Convert to OrderedDict with integer keys
                    def ensure_ordered_dict(feats):
                        if isinstance(feats, dict):
                            return OrderedDict((str(k), v) for k, v in feats.items())
                        elif isinstance(feats, list):
                            expected_keys_list = ['0', '1', '2', '3', 'pool']
                            return OrderedDict((k, f) for k, f in zip(expected_keys_list, feats))
                        elif isinstance(feats, torch.Tensor):
                            return OrderedDict({'0': feats})
                        else:
                            raise TypeError(f"Unsupported feature type: {type(feats)}")

                    # === Force correct format before RPN ===
                    src_feats = ensure_ordered_dict(src_feats_raw)
                    print(f"[CHECK] Type before RPN: {type(src_feats)}")
                    tgt_feats = ensure_ordered_dict(tgt_feats_raw)

                    print(f"[CHECK] Final src_feats type: {type(src_feats)}")
                    print(f"[CHECK] Final src_feats keys: {list(src_feats.keys())}")
                    print(f"[CHECK] First map shape: {list(src_feats.values())[0].shape}")

                    assert isinstance(src_feats, OrderedDict), "src_feats is not an OrderedDict!"
                    assert isinstance(tgt_feats, OrderedDict), "tgt_feats is not an OrderedDict!"

                    print(f"[FINAL CHECK] src_feats type: {type(src_feats)}")
                    print(f"[FINAL CHECK] src_feats keys: {list(src_feats.keys()) if isinstance(src_feats, dict) else 'NOT A DICT'}")

                    expected_keys = ['0', '1', '2', '3', 'pool']
                    assert all(k in src_feats for k in
                               expected_keys), f"Missing keys in features for RPN: expected {expected_keys}, got {list(src_feats.keys())}"

                    print("RPN input check:", type(src_feats), isinstance(src_feats, OrderedDict))
                    # === Call RPN safely ===
                    src_proposals, _ = self.get_proposals_from_rpn(self.detector, source_images, src_feats, targets=source_targets)
                    tgt_proposals, _ = self.get_proposals_from_rpn(self.detector, target_images, tgt_feats, targets=None)

                    # ROI pooling
                    src_box_features = self.detector.roi_heads.box_roi_pool(
                        src_feats, src_proposals, src_image_list.image_sizes
                    )
                    tgt_box_features = self.detector.roi_heads.box_roi_pool(
                        tgt_feats, tgt_proposals, tgt_image_list.image_sizes
                    )

                    # Box head feature extraction
                    src_proposal_feats = self.detector.roi_heads.box_head(src_box_features)
                    tgt_proposal_feats = self.detector.roi_heads.box_head(tgt_box_features)

            except Exception as e:
                import traceback
                print(f"[ERROR] Exception during instance-level domain adaptation:")
                traceback.print_exc()
                print(f"[INFO] Continuing without instance-level domain loss...")
                src_proposal_feats = None
                tgt_proposal_feats = None

            # Calculate instance-level domain loss if we have proposal features
            if (src_proposal_feats is not None and tgt_proposal_feats is not None and
                    hasattr(src_proposal_feats, 'size') and hasattr(tgt_proposal_feats, 'size') and
                    src_proposal_feats.size(0) > 0 and tgt_proposal_feats.size(0) > 0):

                src_inst_preds = self.instance_domain_classifier(src_proposal_feats)
                tgt_inst_preds = self.instance_domain_classifier(tgt_proposal_feats)
                instance_domain_loss = compute_domain_loss(src_inst_preds, tgt_inst_preds, self.device)
            else:
                instance_domain_loss = torch.tensor(0.0, device=self.device)

            # === Total loss & optimization ===
            total_batch_loss = detection_loss + img_domain_loss + instance_domain_loss
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()

            # Update progress bar with loss
            progress_bar.set_postfix({
                "Det_Loss": f"{detection_loss.item():.4f}",
                "Img_Dom_Loss": f"{img_domain_loss.item():.4f}",
                "Inst_Dom_Loss": f"{instance_domain_loss.item():.4f}",
                "Total": f"{total_batch_loss.item():.4f}"
            })

        self.lr_scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"\n[INFO] Average Training Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self, num_epochs=25):
        """Train the model across multiple epochs."""
        source_loader, target_loader = self.get_dataloaders()

        try:
            for epoch in range(num_epochs):
                print(f"\n[INFO] Epoch {epoch + 1}/{num_epochs}")
                self.train_one_epoch(source_loader, target_loader)

                # Save checkpoint every 5 epochs
                if (epoch + 1) % 5 == 0:
                    os.makedirs('experiments/checkpoints', exist_ok=True)
                    torch.save(self.detector.state_dict(),
                               f'experiments/checkpoints/faster_rcnn_da_epoch{epoch + 1}.pth')
                    print(f"[INFO] Saved checkpoint at epoch {epoch + 1}")

        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user. Saving final model...")
            os.makedirs('experiments/checkpoints', exist_ok=True)
            torch.save(self.detector.state_dict(), 'experiments/checkpoints/faster_rcnn_da_interrupted.pth')
        except Exception as e:
            print(f"\n[ERROR] Training failed with exception: {e}")
            import traceback
            traceback.print_exc()
            print("\n[INFO] Attempting to save model before exit...")
            try:
                os.makedirs('experiments/checkpoints', exist_ok=True)
                torch.save(self.detector.state_dict(), 'experiments/checkpoints/faster_rcnn_da_error_recovery.pth')
                print("[INFO] Recovery checkpoint saved.")
            except:
                print("[ERROR] Could not save recovery checkpoint.")
            raise

        print("\n[INFO] Domain-Adaptive Training Complete!")
        return self.detector