# Domain Adaptive Object Detection for Autonomous Driving Under Foggy Weather
This repository contains the implementation of a domain-adaptive object detection framework based on Faster R-CNN for improving object detection performance in foggy weather conditions.

---

## 🚘 Project Overview
Autonomous vehicles face significant challenges when navigating through adverse weather conditions such as fog, which can obscure objects and reduce the reliability of object detection systems. Most detectors are trained under clear weather, causing performance degradation in unseen domains (e.g., fog).

This project implements a domain adaptation approach using Faster R-CNN to bridge the performance gap between clear and foggy weather conditions without requiring additional labeled data in the foggy domain.

We train the model on Cityscapes (clear) images and adapt to Foggy Cityscapes using adversarial learning and domain classifiers.

---

## Environment Setup
```sh
git clone git@github.com:emapendo/Domain-Adaptive-Object-Detection.git
cd domain-adaptive-object-detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt # Install dependencies
```

---

## Dataset Preparation
📦 Download the following files from the Cityscapes website:
**leftImg8bit_trainvaltest.zip 
**gtFine_trainvaltest.zip
**leftImg8bit_trainvaltest_foggy.zip (from Foggy Cityscapes)

```sh
data/
├── cityscapes/
│   ├── leftImg8bit/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── gtFine/
│       ├── train/
│       ├── val/
│       └── test/
└── foggy_cityscapes/
    └── leftImg8bit_foggy/
        ├── train/
        ├── val/
        └── test/
```

---

## 📂 Project Structure
```sh

```

---

## References
Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28.

Kamath, A., Gupta, S., & Carvalho, V. (2019, July). Reversing gradients in adversarial domain adaptation for question deduplication and textual entailment tasks. In Proceedings of the 57th annual meeting of the association for computational linguistics (pp. 5545-5550).

Xu, M., Chen, Z., Zhang, J., Ni, B., & Tian, Q. (2020). Exploring categorical regularization for domain adaptive object detection. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, 9507–9516.

Li, J., Xu, R., Ma, J., Zou, Q., Ma, J., & Yu, H. (2023). Domain adaptive object detection for autonomous driving under foggy weather. In Proceedings of the IEEE/CVF winter conference on applications of computer vision (pp. 612-622).