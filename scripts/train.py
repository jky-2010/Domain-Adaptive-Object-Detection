# Author: Elias Mapendo
# Description: Training loop for Faster R-CNN on Cityscapes object detection

import torch
from torch.utils.data import DataLoader
from models.faster_cnn import get_faster_rcnn_model
from data.datasets import CityscapesDataset
from data.preprocessing import BasicTransform
from tqdm import tqdm  # For live progress bars


def collate_fn(batch):
    """Custom collate function to handle batches of variable-sized annotations."""
    return tuple(zip(*batch))


def train_model(num_epochs=10, batch_size=2, lr=0.005, device='cuda'):
    print("\n[INFO] Starting training loop...")

    # Target label IDs from Cityscapes for object detection
    target_labels = [24, 25, 26, 27, 28, 31, 32, 33]  # person, rider, car, truck, bus, train, motorcycle, bicycle
    num_classes = len(target_labels) + 1  # +1 for background class

    # Data setup with preprocessing and dataset instantiation
    transform = BasicTransform()
    train_dataset = CityscapesDataset(
        mode='train',
        foggy=False,
        transforms=transform,
        target_labels=target_labels
    )
    # Use num_workers=0 for macOS compatibility and debugging
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # Load model with specified number of classes
    model = get_faster_rcnn_model(num_classes=num_classes)
    model.to(device)

    # Set up optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        print(f"\n[INFO] Epoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_loss = 0

        # Wrap DataLoader with tqdm for a real-time progress bar
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} Progress"):
            # Move data to the appropriate device (GPU or CPU)
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass and compute loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        # Step learning rate scheduler
        lr_scheduler.step()
        print(f"[INFO] Epoch Loss: {epoch_loss:.4f}")

    print("\n[INFO] Training completed.")
    return model


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_model = train_model(device=device)
    torch.save(trained_model.state_dict(), 'experiments/faster_rcnn_cityscapes.pth')
