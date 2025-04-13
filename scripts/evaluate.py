# Author: Elias Mapendo
# Description: Evaluate Faster R-CNN on Foggy Cityscapes (target domain) to measure baseline performance

import torch
from torch.utils.data import DataLoader
from models.faster_cnn import get_faster_rcnn_model
from data.datasets import CityscapesDataset
from data.preprocessing import BasicTransform
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def collate_fn(batch):
    return tuple(zip(*batch))


def visualize_predictions(img, boxes, scores, threshold=0.5):
    """Visualize image with predicted bounding boxes."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img.permute(1, 2, 0))

    for i in range(len(boxes)):
        if scores[i] >= threshold:
            box = boxes[i]
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
    plt.title("Predictions on Foggy Image")
    plt.axis('off')
    plt.show()


def evaluate_on_foggy(model_path, device='cuda'):
    print("\n[INFO] Evaluating on Foggy Cityscapes...")

    target_labels = [24, 25, 26, 27, 28, 31, 32, 33]  # person, rider, car, truck, bus, train, motorcycle, bicycle
    num_classes = len(target_labels) + 1

    # Load foggy dataset
    transform = BasicTransform()
    foggy_dataset = CityscapesDataset(
        mode='val',  # using val split for evaluation
        foggy=True,
        transforms=transform,
        target_labels=target_labels
    )
    foggy_loader = DataLoader(foggy_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Load trained model
    model = get_faster_rcnn_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for idx, (images,) in enumerate(foggy_loader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            img = images[0].cpu()
            boxes = outputs[0]['boxes'].cpu()
            scores = outputs[0]['scores'].cpu()

            visualize_predictions(img, boxes, scores)

            if idx == 4:  # visualize 5 images only
                break

    print("\n[INFO] Evaluation complete.")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'experiments/faster_rcnn_cityscapes.pth'
    evaluate_on_foggy(model_path, device=device)