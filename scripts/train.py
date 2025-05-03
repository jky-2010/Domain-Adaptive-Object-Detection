"""
Author: Elias Mapendo
Date: May 03, 2025
Description:
Training loop for Faster R-CNN on Cityscapes object detection with checkpoint saving and resuming.
"""

import torch, os
from torch.utils.data import DataLoader, Subset
from models.faster_cnn import get_faster_rcnn_model
from data.datasets import CityscapesDataset
from data.preprocessing import BasicTransform
from utils.checkpoints import save_checkpoint, load_checkpoint
from tqdm import tqdm
from data.utils import collate_fn


def train_model(num_epochs=10, batch_size=2, lr=0.005, device='cuda', resume_epoch=0):
    print("\n[INFO] Starting training loop...")

    target_labels = [24, 25, 26, 27, 28, 31, 32, 33]
    num_classes = len(target_labels) + 1

    transform = BasicTransform()
    full_dataset = CityscapesDataset(mode='train', foggy=False, transforms=transform, target_labels=target_labels)
    subset_size = int(len(full_dataset) * 0.8)
    train_dataset = Subset(full_dataset, list(range(subset_size)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model = get_faster_rcnn_model(num_classes=num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    os.makedirs('experiments', exist_ok=True)

    if resume_epoch > 0:
        checkpoint_path = f"experiments/faster_rcnn_epoch_{resume_epoch}.pth"
        if os.path.exists(checkpoint_path):
            print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
            resume_epoch = load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path)
        else:
            print(f"[WARNING] Checkpoint not found: {checkpoint_path}. Starting from scratch.")
            resume_epoch = 0

    for epoch in range(resume_epoch, num_epochs):
        print(f"\n[INFO] Epoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_loss = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} Progress"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()
        print(f"[INFO] Epoch Loss: {epoch_loss:.4f}")

        checkpoint_path = f"experiments/faster_rcnn_epoch_{epoch+1}.pth"
        save_checkpoint(model, optimizer, lr_scheduler, epoch+1, checkpoint_path)
        print(f"[INFO] Saved checkpoint to {checkpoint_path}")

    print("\n[INFO] Training completed.")
    return model


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resume_from_epoch = 0
    trained_model = train_model(device=device, num_epochs=25, resume_epoch=resume_from_epoch)
    torch.save(trained_model.state_dict(), 'experiments/faster_rcnn_cityscapes.pth')
