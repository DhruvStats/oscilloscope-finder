"""Run the full pipeline (detect + brand + OCR model number) on an image or a folder.

For each input image this writes, into outputs/:
    <name>_result.jpg   image with boxes and labels drawn
    <name>_result.txt   brand, model number, confidences, box coordinates

    python scripts/predict.py path/to/photo.jpg
    python scripts/predict.py data/yolo/images/test
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oscilloscope_pipeline import OscilloscopeAnalyzer, annotate, to_text  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="image file or folder of images")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--conf", type=float, default=0.25, help="minimum detection confidence")
    ap.add_argument("--no-ocr", action="store_true", help="skip model-number reading")
    args = ap.parse_args()

    src = Path(args.source)
    images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS) if src.is_dir() else [src]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    analyzer = OscilloscopeAnalyzer(use_ocr=not args.no_ocr)
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"could not read {path}, skipped")
            continue
        result = analyzer.analyze(image, path.name, conf=args.conf)
        cv2.imwrite(str(out / f"{path.stem}_result.jpg"), annotate(image, result))
        (out / f"{path.stem}_result.txt").write_text(to_text(result))

        summary = ", ".join(f"{d.brand} {d.model_number} ({d.confidence:.2f}, via {d.model_source})"
                            for d in result.detections) or "no oscilloscope found"
        print(f"{path.name}: {summary}")
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
