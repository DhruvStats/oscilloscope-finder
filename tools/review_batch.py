"""Candidate boxes for a batch of new images, as numbered review sheets.

    .venv/Scripts/python tools/review_batch.py raw/batch_2026-10-08/photos raw/batch_2026-10-08/frames_v02 ...

For every image the current model proposes all boxes down to 5% confidence (whole photo + tiles). Each
candidate gets a number on the sheet; a reviewer then decides per image which numbers to keep and with which
class (see tools/apply_review.py). Writes raw/review/candidates.json and raw/review/sheet_NN.jpg.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))
os.environ.setdefault("CONTEXT_MODEL", "off")
from app import COLOURS, TARGET  # noqa: E402

PER_SHEET = 12
TILE = 440


def hex_bgr(h):
    return tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))


def main():
    folders = sys.argv[1:]
    out_dir = os.path.join(ROOT, "raw", "review")
    os.makedirs(out_dir, exist_ok=True)
    items, tiles = [], []
    files = [p for f in folders for p in sorted(glob.glob(os.path.join(ROOT, f, "*.jpg")))]
    for k, path in enumerate(files):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        cands = []
        for d in TARGET.detect(img, 0.05):
            b = d["bbox"]
            cands.append({"cls": d["label"], "conf": round(d["confidence"], 3),
                          "box": [round(b["x"] * w), round(b["y"] * h), round((b["x"] + b["width"]) * w),
                                  round((b["y"] + b["height"]) * h)]})
        cands.sort(key=lambda c: -c["conf"])
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        items.append({"id": k, "file": rel, "w": w, "h": h, "candidates": cands})

        vis = img.copy()
        lw = max(2, w // 400)
        for i, c in enumerate(cands):
            x0, y0, x1, y1 = c["box"]
            col = hex_bgr(COLOURS.get(c["cls"], "#18c27f")) if c["conf"] >= 0.3 else (160, 160, 160)
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, lw * (2 if c["conf"] >= 0.3 else 1))
            tag = f"{i}:{c['cls'][4:]} {int(c['conf'] * 100)}"
            fs = max(0.6, w / 1100)
            (tw, th), _ = cv2.getTextSize(tag, 0, fs, 2)
            ty = max(th + 6, y0)
            cv2.rectangle(vis, (x0, ty - th - 6), (x0 + tw + 6, ty), col, -1)
            cv2.putText(vis, tag, (x0 + 3, ty - 4), 0, fs, (0, 0, 0), 2, cv2.LINE_AA)
        s = TILE / max(h, w)
        t = np.full((TILE + 22, TILE, 3), 255, np.uint8)
        r = cv2.resize(vis, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        t[:r.shape[0], :r.shape[1]] = r
        cv2.putText(t, f"#{k} {rel[-34:]}", (3, TILE + 16), 0, 0.42, (0, 0, 0), 1)
        tiles.append(t)
        print(f"{k:>3} {rel}  {len(cands)} candidates", flush=True)

    with open(os.path.join(out_dir, "candidates.json"), "w") as f:
        json.dump(items, f, indent=1)
    for s in range(0, len(tiles), PER_SHEET):
        chunk = tiles[s:s + PER_SHEET]
        while len(chunk) % 4:
            chunk.append(np.full((TILE + 22, TILE, 3), 255, np.uint8))
        sheet = np.vstack([np.hstack(chunk[i:i + 4]) for i in range(0, len(chunk), 4)])
        cv2.imwrite(os.path.join(out_dir, f"sheet_{s // PER_SHEET:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(items)} images, {sum(len(i['candidates']) for i in items)} candidates, "
          f"{(len(tiles) + PER_SHEET - 1) // PER_SHEET} sheets in raw/review/")


if __name__ == "__main__":
    main()
