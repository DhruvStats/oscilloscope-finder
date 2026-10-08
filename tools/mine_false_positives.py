"""Hard-negative mining: find what the detector wrongly calls a target in the labelled TRAIN images.

Runs the deployed model (whole photo + tiles) over every real train image in raw/real_labels.json; every box
with confidence >= --conf that does not overlap a labelled target is a false positive. Those crops are saved
to synth/hard_negatives/ and pasted, unlabelled, into new training scenes by synth/gen3.py, so the next model
learns that speakers, keyboards, monitors etc. are not targets. Test images are never used.

    .venv/Scripts/python tools/mine_false_positives.py [--conf 0.25]
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "synth", "hard_negatives")


def overlap(a, b):
    """Fraction of box a (x0,y0,x1,y1) covered by box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, (a[2] - a[0]) * (a[3] - a[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    os.environ.setdefault("CONTEXT_MODEL", "off")
    os.environ.setdefault("TEK_CHECK", "off")
    sys.path.insert(0, os.path.join(ROOT, "server"))
    from app import TARGET

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    labels = json.load(open(os.path.join(ROOT, "raw", "real_labels.json")))["images"]
    found, tiles = [], []
    for it in labels:
        if it.get("split") == "test":
            continue
        img = cv2.imread(os.path.join(ROOT, it["file"]))
        H, W = img.shape[:2]
        gts = [[b["bbox"][0], b["bbox"][1], b["bbox"][0] + b["bbox"][2], b["bbox"][1] + b["bbox"][3]]
               for b in it["boxes"]]
        for d in TARGET.detect(img, args.conf):
            b = d["bbox"]
            box = [int(b["x"] * W), int(b["y"] * H), int((b["x"] + b["width"]) * W), int((b["y"] + b["height"]) * H)]
            if (box[2] - box[0]) < 12 or (box[3] - box[1]) < 12:
                continue
            # part of / around a real scope = localisation issue, not a false object
            if any(overlap(box, g) > 0.3 or overlap(g, box) > 0.3 for g in gts):
                continue
            crop = img[box[1]:box[3], box[0]:box[2]]
            name = f"{len(found):04d}_{d['confidence']:.2f}_{os.path.basename(it['file'])}"
            cv2.imwrite(os.path.join(OUT, name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            found.append({"crop": name, "file": it["file"], "box": box, "conf": round(d["confidence"], 3),
                          "said": d["label"]})
            s = 120 / max(crop.shape[:2])
            t = np.full((132, 120, 3), 255, np.uint8)
            r = cv2.resize(crop, None, fx=s, fy=s)
            t[:r.shape[0], :r.shape[1]] = r
            cv2.putText(t, f"{len(found) - 1} {d['confidence']:.2f}", (2, 129), 0, 0.38, (0, 0, 255), 1)
            tiles.append(t)
    json.dump(found, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    if tiles:
        while len(tiles) % 14:
            tiles.append(np.full((132, 120, 3), 255, np.uint8))
        cv2.imwrite(os.path.join(OUT, "_sheet.jpg"), np.vstack([np.hstack(tiles[i:i + 14])
                                                                for i in range(0, len(tiles), 14)]))
    print(f"{len(found)} false positives (conf >= {args.conf}) in "
          f"{sum(1 for l in labels if l.get('split') != 'test')} train images -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
