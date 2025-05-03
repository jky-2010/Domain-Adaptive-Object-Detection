"""
Author: Elias Mapendo
Date: April 25, 2025
Description:
Training script to launch domain-adaptive object detection training.
Uses clear (source) and foggy (target) Cityscapes datasets to adapt Faster R-CNN for robust detection under adverse weather.
"""

import sys, os, torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers.domain_adaptive_trainer import DomainAdaptiveTrainer

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    trainer = DomainAdaptiveTrainer(
        device=device,
        batch_size=2,
        lr=0.005
    )

    trained_detector = trainer.train(num_epochs=5)

    # Save adapted model
    os.makedirs('experiments', exist_ok=True)
    torch.save(trained_detector.state_dict(), 'experiments/faster_rcnn_domain_adapted.pth')
    print("[INFO] Saved domain-adapted model to 'experiments/faster_rcnn_domain_adapted.pth'")

if __name__ == "__main__":
    main()