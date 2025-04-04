import torch
import numpy as np

# Author: Elias Mapendo
# Description: Converts segmentation masks to bounding boxes for object detection

def masks_to_boxes(segmentation_mask, target_labels):
    """
    Convert a segmentation mask into bounding boxes for target object classes.

    Args:
        segmentation_mask (Tensor): A (H, W) mask where each pixel has a class label.
        target_labels (list of int): Raw label IDs from Cityscapes (e.g., 24, 25...).

    Returns:
        boxes (Tensor[N, 4]): Bounding boxes in [xmin, ymin, xmax, ymax] format.
        labels (Tensor[N]): Class labels remapped to 0...N.
    """
    boxes = []
    labels = []

    # Map raw label IDs to class indices
    label_id_to_class_index = {raw_id: i for i, raw_id in enumerate(target_labels)}

    for raw_id in target_labels:
        mask = (segmentation_mask == raw_id).numpy().astype(np.uint8)
        if mask.sum() == 0:
            continue

        from scipy.ndimage import label, find_objects
        labeled, num = label(mask)
        slices = find_objects(labeled)

        for sl in slices:
            y_min, x_min = sl[0].start, sl[1].start
            y_max, x_max = sl[0].stop, sl[1].stop
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label_id_to_class_index[raw_id])  # remapped class index

    if boxes:
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
    else:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
