"""Score a checkpoint the way the app uses it (whole photo + tiles, merged) on the real test photos.

    .venv/Scripts/python tools/eval_app.py models/deploy/yolox_tiny_osc3.pth [--conf 0.4]

For every real scope: found with the right name / found with the wrong name / missed; plus false alarms.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--conf", type=float, default=0.4)
    args = ap.parse_args()
    os.environ["TARGET_CKPT"] = os.path.abspath(args.ckpt)
    os.environ["TARGET_EXP"] = "yolox_tiny_osc3"
    os.environ["CONTEXT_MODEL"] = "off"
    sys.path.insert(0, os.path.join(ROOT, "server"))
    import cv2
    from app import TARGET

    d = os.path.join(ROOT, "datasets", "oscilloscopes3")
    gt = json.load(open(os.path.join(d, "annotations", "instances_test2017.json")))
    names = {c["id"]: c["name"] for c in gt["categories"]}
    right = wrong = missed = false = 0
    for im in gt["images"]:
        img = cv2.imread(os.path.join(d, "test2017", im["file_name"]))
        h, w = img.shape[:2]
        preds = []
        for p in TARGET.detect(img, 0.05):
            if p["confidence"] < args.conf:
                continue
            b = p["bbox"]
            preds.append(([b["x"] * w, b["y"] * h, (b["x"] + b["width"]) * w, (b["y"] + b["height"]) * h], p["label"]))
        used, line = set(), []
        for a in [a for a in gt["annotations"] if a["image_id"] == im["id"]]:
            x, y, bw, bh = a["bbox"]
            box, truth = [x, y, x + bw, y + bh], names[a["category_id"]]
            cand = [(iou(box, p[0]), i) for i, p in enumerate(preds) if i not in used]
            best = max(cand, default=(0, -1))
            if best[0] >= 0.4:
                used.add(best[1])
                if preds[best[1]][1] == truth:
                    right += 1
                    line.append(f"{truth[4:]} ok")
                else:
                    wrong += 1
                    line.append(f"{truth[4:]} as {preds[best[1]][1][4:]}")
            else:
                missed += 1
                line.append(f"{truth[4:]} MISSED")
        fa = len(preds) - len(used)
        false += fa
        print(f"{im['file_name']:<26} {', '.join(line)}{f'  +{fa} false alarm(s)' if fa else ''}")
    total = right + wrong + missed
    print(f"\nright {right}/{total}   wrong name {wrong}   missed {missed}   false alarms {false}")


if __name__ == "__main__":
    main()
