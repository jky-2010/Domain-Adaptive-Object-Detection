# Author: James Yang
# Description: Initializes a Faster R-CNN model with configurable backbone and head for object detection on Cityscapes

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


def get_faster_rcnn_model(num_classes=8, pretrained=True):
    """
    Returns a Faster R-CNN model with a ResNet-50 backbone and custom head.

    Args:
        num_classes (int): Number of output classes including background.
        pretrained (bool): Whether to use a model pretrained on COCO.

    Returns:
        model (nn.Module): Faster R-CNN object detection model.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None

    # Load a pre-trained model for classification and return
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Get the input dimension of the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the head with a new one for our dataset
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
