"""Close-up crops of review candidates, so the instrument model can be checked (e.g. TDS 2014 vs TDS 1002).

    .venv/Scripts/python tools/review_crops.py 0 38 [--min-conf 0.1]

Reads raw/review/candidates.json, writes raw/review/crops_<first>_<last>_NN.jpg: one tile per candidate,
labelled "#image:candidate class conf".
"""
import argparse
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--min-conf", type=float, default=0.1)
    ap.add_argument("--per-sheet", type=int, default=24)
    args = ap.parse_args()
    items = json.load(open(os.path.join(ROOT, "raw", "review", "candidates.json")))
    tiles = []
    for it in items[args.first:args.last + 1]:
        img = cv2.imread(os.path.join(ROOT, it["file"]))
        h, w = img.shape[:2]
        for i, c in enumerate(it["candidates"]):
            if c["conf"] < args.min_conf:
                continue
            x0, y0, x1, y1 = c["box"]
            px, py = int((x1 - x0) * 0.15) + 10, int((y1 - y0) * 0.15) + 10
            X0, Y0, X1, Y1 = max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)
            crop = img[Y0:Y1, X0:X1].copy()
            cv2.rectangle(crop, (x0 - X0, y0 - Y0), (x1 - X0, y1 - Y0), (0, 0, 255), 2)
            s = T / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
            t = np.full((T + 20, T, 3), 255, np.uint8)
            t[:crop.shape[0], :crop.shape[1]] = crop
            cv2.putText(t, f"#{it['id']}:{i} {c['cls'][4:]} {int(c['conf'] * 100)}", (3, T + 15), 0, 0.5,
                        (0, 0, 0), 1)
            tiles.append(t)
    for s in range(0, len(tiles), args.per_sheet):
        chunk = tiles[s:s + args.per_sheet]
        while len(chunk) % 6:
            chunk.append(np.full((T + 20, T, 3), 255, np.uint8))
        out = os.path.join(ROOT, "raw", "review", f"crops_{args.first}_{args.last}_{s // args.per_sheet:02d}.jpg")
        cv2.imwrite(out, np.vstack([np.hstack(chunk[i:i + 6]) for i in range(0, len(chunk), 6)]))
        print(out)


if __name__ == "__main__":
    main()
