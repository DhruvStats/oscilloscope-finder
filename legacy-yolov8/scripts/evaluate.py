"""Evaluate models/best.pt on the held-out REAL test photos (data/yolo/images/test).

Reports precision, recall and mAP, and saves the test photos with predicted boxes drawn
to runs/test_predictions/ so you can show them in your report.

    python scripts/evaluate.py
"""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    weights = ROOT / "models" / "best.pt"
    if not weights.exists():
        raise SystemExit("models/best.pt not found - run  python scripts/train.py  first")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(ROOT / "data" / "yolo" / "data.yaml"),
        split="test",
        project=str(ROOT / "runs"),
        name="test_metrics",
        exist_ok=True,
    )
    print("\nReal-photo test set")
    print(f"  precision   {metrics.box.mp:.3f}")
    print(f"  recall      {metrics.box.mr:.3f}")
    print(f"  mAP@0.5     {metrics.box.map50:.3f}")
    print(f"  mAP@0.5:.95 {metrics.box.map:.3f}")

    model.predict(
        source=str(ROOT / "data" / "yolo" / "images" / "test"),
        project=str(ROOT / "runs"),
        name="test_predictions",
        exist_ok=True,
        save=True,
        conf=0.25,
    )
    print(f"\nAnnotated test photos saved to {ROOT / 'runs' / 'test_predictions'}")


if __name__ == "__main__":
    main()
