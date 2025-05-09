# Author: James Yang
# Description: Evaluate a trained Faster R-CNN model on paired Clear and Foggy Cityscapes images using filenames

import sys, os, torch, csv, random, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from models.faster_cnn import get_faster_rcnn_model
from data.datasets import CityscapesDataset
from data.preprocessing import BasicTransform
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.ops import box_iou
import matplotlib.pyplot as plt


def collate_fn(batch):
    return tuple(zip(*batch)) if isinstance(batch[0], tuple) else batch


def visualize_and_save_predictions(images, outputs, output_dir, prefix, threshold=0.5, targets=None):
    os.makedirs(output_dir, exist_ok=True)
    for i, (img, output) in enumerate(zip(images, outputs)):
        pred_boxes = output['boxes'][output['scores'] > threshold].cpu()
        pred_labels = output['labels'][output['scores'] > threshold].cpu()
        drawn = draw_bounding_boxes((img.cpu() * 255).byte(), pred_boxes, labels=[str(l.item()) for l in pred_labels], colors="red")

        if targets is not None:
            gt_boxes = targets[i]['boxes'].cpu()
            drawn = draw_bounding_boxes(drawn, gt_boxes, colors="green")

        img_pil = to_pil_image(drawn)
        img_path = os.path.join(output_dir, f"{prefix}.png")
        img_pil.save(img_path)


def compute_iou(gt_boxes, pred_boxes, threshold=0.5):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0
    ious = box_iou(gt_boxes, pred_boxes)
    matches = (ious > threshold).sum().item()
    return matches / max(len(gt_boxes), 1)


def filter_by_names(dataset, name_list):
    matched_image_paths = []
    matched_anno_paths = []
    matched_names = []

    for name in name_list:
        for img_path in dataset.images:
            if name in os.path.basename(img_path):
                matched_image_paths.append(img_path)
                matched_names.append(name)
                break

        if dataset.annotations_available:
            for ann_path in dataset.annotations:
                if name in os.path.basename(ann_path):
                    matched_anno_paths.append(ann_path)
                    break

    dataset.images = matched_image_paths
    if dataset.annotations_available:
        dataset.annotations = matched_anno_paths

    return dataset, matched_names


def evaluate_on_split(model, split, foggy, device, image_names):
    name = "Foggy" if foggy else "Clear"
    print(f"\n[INFO] Evaluating on {name} Cityscapes ({split})...")

    target_labels = [24, 25, 26, 27, 28, 31, 32, 33]
    transform = BasicTransform()
    dataset = CityscapesDataset(mode=split, foggy=foggy, transforms=transform, target_labels=target_labels)
    dataset, filtered_names = filter_by_names(dataset, image_names)

    if len(filtered_names) == 0:
        print("[ERROR] No matching images found.")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model.eval()
    total_iou = 0.0
    imagewise_iou = []
    output_dir = f"outputs/{name.lower()}"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= len(filtered_names): break
            filename = filtered_names[idx]

            if foggy:
                images = [batch[0].to(device)] if isinstance(batch, tuple) else [b.to(device) for b in batch]
                targets = None
            else:
                images, targets = batch
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            visualize_and_save_predictions(images, outputs, output_dir, prefix=f"{name.lower()}_{filename}", targets=targets if not foggy else None)

            if not foggy:
                gt_boxes = targets[0]['boxes']
                pred_boxes = outputs[0]['boxes'][outputs[0]['scores'] > 0.5]
                iou = compute_iou(gt_boxes, pred_boxes)
                total_iou += iou
                imagewise_iou.append((filename, iou))

    if not foggy:
        avg_iou = total_iou / len(imagewise_iou)
        print(f"[INFO] Average IoU on {name} set: {avg_iou:.4f}")

        csv_path = os.path.join(output_dir, f"{name.lower()}_iou_report.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image', 'IoU'])
            writer.writerows(imagewise_iou)
            writer.writerow(['Average', avg_iou])

        plt.figure(figsize=(10, 5))
        plt.bar([x[0] for x in imagewise_iou], [x[1] for x in imagewise_iou], color='skyblue')
        plt.axhline(avg_iou, color='red', linestyle='--', label=f'Average IoU: {avg_iou:.2f}')
        plt.xlabel('Image')
        plt.ylabel('IoU')
        plt.title(f'{name} Image-wise IoU')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name.lower()}_iou_chart.png"))
        plt.close()
    else:
        print("[WARN] Skipping IoU: annotations not available for foggy set.")


def evaluate_model(model_path, device='cuda', shared_names_path="outputs/eval_names.json"):
    target_labels = [24, 25, 26, 27, 28, 31, 32, 33]
    num_classes = len(target_labels) + 1

    model = get_faster_rcnn_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    transform = BasicTransform()
    full_dataset = CityscapesDataset(mode='val', foggy=False, transforms=transform, target_labels=target_labels)
    image_names = [os.path.basename(p).replace('_leftImg8bit.png', '') for p in full_dataset.images]
    num_images = 50

    if os.path.exists(shared_names_path):
        with open(shared_names_path, 'r') as f:
            selected_names = json.load(f)
    else:
        selected_names = random.sample(image_names, num_images)
        with open(shared_names_path, 'w') as f:
            json.dump(selected_names, f)

    evaluate_on_split(model, split='val', foggy=False, device=device, image_names=selected_names)
    evaluate_on_split(model, split='val', foggy=True, device=device, image_names=selected_names)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evaluate_model(
        model_path=os.path.join(project_root, "experiments", "faster_rcnn_cityscapes.pth"),
        device=device
    )
