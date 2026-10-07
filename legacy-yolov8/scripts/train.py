"""Train a YOLOv8 detector on data/yolo and copy the best weights to models/best.pt.

YOLO augments the training images on the fly every epoch (mosaic, scale, translation,
colour/brightness changes), so each of the ~140 training images is seen in many variations.
Horizontal flip is turned off because it would mirror the brand logo and model text.

    python scripts/train.py                 # defaults: yolov8n, 60 epochs, CPU or GPU if present
    python scripts/train.py --epochs 100 --model yolov8s.pt
"""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained starting weights")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="lower this if you run out of RAM")
    ap.add_argument("--workers", type=int, default=2, help="data-loading processes (each uses RAM)")
    args = ap.parse_args()

    data = ROOT / "data" / "yolo" / "data.yaml"
    if not data.exists():
        raise SystemExit("data/yolo/data.yaml not found - run  python scripts/prepare_dataset.py  first")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on {'GPU' if device == 0 else 'CPU'}")

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(ROOT / "runs"),
        name="detect",
        exist_ok=True,
        patience=20,      # stop early if validation mAP stops improving
        fliplr=0.0,       # never mirror: logos and model numbers must stay readable
        seed=42,
        plots=True,
    )

    best = ROOT / "runs" / "detect" / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy2(best, ROOT / "models" / "best.pt")
    print(f"\nBest weights copied to {ROOT / 'models' / 'best.pt'}")
    print("Training curves and confusion matrix: runs/detect/")


if __name__ == "__main__":
    main()
