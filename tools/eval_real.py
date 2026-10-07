"""Score a checkpoint on the real held-out photos (datasets/oscilloscopes3/test2017).

Prints COCO AP per class plus a confusion table: for every real scope, what the model said
(the right model, another model, or nothing) at IoU >= 0.5 and confidence >= --conf.

    .venv/Scripts/python tools/eval_real.py models/training/yolox_tiny_osc3/best_ckpt.pth
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models", "yolox"))
sys.path.insert(0, os.path.join(ROOT, "training"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from yolox.data.data_augment import ValTransform  # noqa: E402
from yolox.utils import postprocess  # noqa: E402

from train_cpu import evaluate  # noqa: E402
from yolox_tiny_osc3 import Exp  # noqa: E402


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / max(1e-6, a[2] * a[3] + (bx2 - bx1) * (by2 - by1) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()
    exp = Exp()
    model = exp.get_model().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])

    print("== COCO AP on real held-out photos ==")
    evaluate(model, exp, "test")

    with open(os.path.join(exp.data_dir, "annotations", "instances_test2017.json")) as f:
        gt = json.load(f)
    names = [c["name"] for c in sorted(gt["categories"], key=lambda c: c["id"])]
    conf = {n: {m: 0 for m in names + ["missed"]} for n in names}
    false_alarms = 0
    tf = ValTransform(legacy=False)
    model.eval()
    for im in gt["images"]:
        img = cv2.imread(os.path.join(exp.data_dir, "test2017", im["file_name"]))
        ratio = min(exp.test_size[0] / img.shape[0], exp.test_size[1] / img.shape[1])
        x, _ = tf(img, None, exp.test_size)
        with torch.no_grad():
            out = postprocess(model(torch.from_numpy(x).unsqueeze(0).float()), exp.num_classes,
                              args.conf, exp.nmsthre, True)[0]
        preds = [] if out is None else [(np.array(p[:4]) / ratio, int(p[6]), float(p[4] * p[5])) for p in out.numpy()]
        used = set()
        for a in [a for a in gt["annotations"] if a["image_id"] == im["id"]]:
            best = max(((iou(a["bbox"], p[0]), i) for i, p in enumerate(preds) if i not in used), default=(0, -1))
            truth = names[a["category_id"] - 1]
            if best[0] >= 0.5:
                used.add(best[1])
                conf[truth][names[preds[best[1]][1]]] += 1
            else:
                conf[truth]["missed"] += 1
        false_alarms += len(preds) - len(used)

    print(f"\n== Confusion at IoU>=0.5, confidence>={args.conf} (rows = real scope, columns = model said) ==")
    print(f"{'':<14}" + "".join(f"{m:>14}" for m in names + ["missed"]))
    for n in names:
        print(f"{n:<14}" + "".join(f"{conf[n][m]:>14}" for m in names + ["missed"]))
    total = sum(sum(v.values()) for v in conf.values())
    correct = sum(conf[n][n] for n in names)
    print(f"\ncorrectly found and named: {correct}/{total}; false alarms: {false_alarms}")


if __name__ == "__main__":
    main()
