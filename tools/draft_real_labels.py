"""Draft the boxes on the real lab photos (raw/photos) for review.

The model proposes boxes; the manifest (raw/photos_manifest.csv) says which instruments are really in each
photo, so a Tektronix box in a photo that only contains the TDS 1002 becomes tek_tds1002, and a box of a
class the manifest does not list is flagged. The output is a draft: review it, then save as raw/real_labels.json.

    .venv/Scripts/python tools/draft_real_labels.py
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))
os.environ.setdefault("TARGET_MIN_CONF", "0.2")
from app import TARGET  # noqa: E402

# held out for the final score: never used for textures, backgrounds or cut-outs; covers every view
TEST = {"02", "17", "20", "24",               # RTB2004 (front, scenes with other scopes)
        "05", "30", "31", "55", "61",         # TDS 2014 (front, side, back, scene)
        "11", "14", "39", "44", "45"}         # TDS 1002 (back, front, front, back with GPIB, side)
COLOURS = {"rs_rtb2004": (0, 200, 90), "tek_tds2014": (0, 140, 255), "tek_tds1002": (210, 75, 210)}


def refine(img, bbox, grow):
    """Grow the model's box, segment the instrument inside it with GrabCut, return the box around the segment.

    The synthetic-trained model often boxes only the part it recognises (handle, port module); the object
    itself is one contiguous region, so GrabCut recovers its full extent.
    """
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    x0, y0 = max(0, int(x - grow * bw)), max(0, int(y - grow * bh))
    x1, y1 = min(w - 1, int(x + bw + grow * bw)), min(h - 1, int(y + bh + grow * bh))
    s = min(1.0, 700 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mask = np.zeros(small.shape[:2], np.uint8)
    rect = (int(x0 * s), int(y0 * s), max(2, int((x1 - x0) * s)), max(2, int((y1 - y0) * s)))
    # the model's own box is certainly object
    mask[:] = cv2.GC_BGD
    mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = cv2.GC_PR_BGD
    cx0, cy0 = int((x + bw * 0.3) * s), int((y + bh * 0.3) * s)
    cx1, cy1 = int((x + bw * 0.7) * s), int((y + bh * 0.7) * s)
    mask[int(y * s):int((y + bh) * s), int(x * s):int((x + bw) * s)] = cv2.GC_PR_FGD
    mask[cy0:cy1, cx0:cx1] = cv2.GC_FGD
    try:
        cv2.grabCut(small, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return bbox
    m = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = lab[(cy0 + cy1) // 2, (cx0 + cx1) // 2]
    if k == 0:
        return bbox
    rx, ry, rw, rh = st[k, :4] / s
    area, old = rw * rh, bw * bh
    return [round(rx), round(ry), round(rw), round(rh)] if 0.7 * old <= area <= 3.5 * old else bbox


def main():
    with open(os.path.join(ROOT, "raw", "photos_manifest.csv")) as f:
        manifest = {r["photo"]: r for r in csv.DictReader(f)}
    items, tiles = [], []
    for photo, row in manifest.items():
        path = f"raw/photos/{photo}.jpg"
        img = cv2.imread(os.path.join(ROOT, path))
        h, w = img.shape[:2]
        present = set(row["instruments"].split(";"))
        known_tek = [c for c in present if c in ("tek_tds2014", "tek_tds1002")]
        boxes, flags = [], []
        dets = TARGET(img, 0.2)
        expected = [c for c in present if c in COLOURS]
        if len(expected) == 1 and not ({"multiple", "tek_unknown"} & present):
            # exactly one scope in the photo: the model's best box is it, whatever class it guessed
            if dets:
                d = max(dets, key=lambda d: d["confidence"])
                if d["label"] != expected[0]:
                    flags.append(f"{d['label']}->{expected[0]}")
                d = dict(d, label=expected[0])
                dets = [d]
        for d in dets:
            cls = d["label"]
            if cls.startswith("tek_") and len(known_tek) == 1 and cls != known_tek[0]:
                flags.append(f"{cls}->{known_tek[0]}")
                cls = known_tek[0]
            if cls not in present and not ({"multiple", "tek_unknown"} & present):
                flags.append(f"unexpected {cls} {d['confidence']:.2f}")
                continue
            b = d["bbox"]
            box = [round(b["x"] * w), round(b["y"] * h), round(b["width"] * w), round(b["height"] * h)]
            # grow generously when the scope is alone in the photo, carefully when others are nearby
            box = refine(img, box, 0.6 if len(expected) == 1 else 0.15)
            boxes.append({"cls": cls, "bbox": box, "score": d["confidence"]})
        missing =[c for c in expected if c not in {b["cls"] for b in boxes}]
        if missing:
            flags.append("missing " + ",".join(missing))
        items.append({"file": path, "session": row["session"], "split": "test" if photo in TEST else "train",
                      "view": row["view"], "boxes": boxes, "flags": flags})

        vis = img.copy()
        lw = max(3, w // 250)
        for b in boxes:
            x, y, bw, bh = b["bbox"]
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), COLOURS[b["cls"]], lw)
            cv2.putText(vis, f'{b["cls"][4:]} {b["score"]:.2f}', (x + 6, y + 40), 0, w / 900, COLOURS[b["cls"]], lw)
        s = 300 / max(h, w)
        t = np.full((330, 300, 3), 255, np.uint8)
        r = cv2.resize(vis, (int(w * s), int(h * s)))
        t[:r.shape[0], :r.shape[1]] = r
        label = f'{photo} {"TEST " if photo in TEST else ""}{"! " + "; ".join(flags) if flags else ""}'
        cv2.putText(t, label[:48], (3, 318), 0, 0.38, (0, 0, 220) if flags else (0, 0, 0), 1)
        tiles.append(t)

    with open(os.path.join(ROOT, "raw", "real_labels_draft.json"), "w") as f:
        json.dump({"images": items}, f, indent=1)
    while len(tiles) % 8:
        tiles.append(np.full((330, 300, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    cv2.imwrite(os.path.join(ROOT, "raw", "review_a.jpg"), np.vstack(rows[:4]))
    cv2.imwrite(os.path.join(ROOT, "raw", "review_b.jpg"), np.vstack(rows[4:]))
    n_flag = sum(1 for it in items if it["flags"])
    print(f"{len(items)} photos, {sum(len(it['boxes']) for it in items)} boxes, {n_flag} flagged for review")


if __name__ == "__main__":
    main()
