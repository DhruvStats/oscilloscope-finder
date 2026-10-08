"""Turn the reviewed candidates of the 2026-10-08 batch into labels (appended to raw/real_labels.json).

Review done on 2026-10-08 from raw/review/sheet_*.jpg and close-up crops. Identity rules, checked on the
front panels (TDS 2014: coloured channel buttons, 5 BNC; TDS 1002: grey buttons, 3 BNC; port module at back):
  * printer/whiteboard corner: Tek on the boxes (black bar in front) = TDS 2014, Tek on the keyboard with cups
    = TDS 1002
  * dark-desk room = TDS 2014;  ARCADIA bench (multimeters, QR poster) = TDS 1002
  * video v02 (white table): the Tek has the port module, green stickers, grey buttons = TDS 1002
  * video v01 (office): front-facing Tek = TDS 1002 (grey buttons, 3 BNC), the Tek lying on its back = TDS 2014
    (its coloured front shows when it is picked up, frames 256-263)
  * video v00 (other lab, OPTIKA bag): Tek fronts have coloured buttons = TDS 2014
v00 is a room never seen in training: every 4th frame becomes a test image, the rest is left out, so the
score shows how the model does in a new place.

    .venv/Scripts/python tools/apply_review.py
"""
import json
import os
import shutil

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, T14, T02 = "rs_rtb2004", "tek_tds2014", "tek_tds1002"

# photos: image id -> list of (candidate index, class) or (class, x0, y0, x1, y1); None = leave out
PHOTOS = {
    0: [(0, T14), (1, T02), (2, R)], 1: [(0, R), (1, T14)], 2: [(0, R)], 3: [(0, R), (1, T14)], 4: [(0, T14)],
    5: None, 6: [(0, R), (1, T14), (T02, 320, 263, 557, 366)], 7: [(0, T14), (2, T02), (3, R)],
    8: [(0, R), (1, T14)], 9: [(3, T14), (4, R)], 10: [(0, R), (1, T14), (2, T02)], 11: [(0, R)],
    12: [(0, R), (1, T14)], 13: [(0, R)], 14: [(0, R), (1, T14)], 15: [(0, T14)],
    16: [(2, R), (1, T02), (3, T14)], 17: [(0, T02)], 18: [(0, R), (4, T02)], 19: [(0, R)], 20: [(0, R)],
    21: [(0, T02)], 22: [(1, R)], 23: [(0, T02)], 24: [(0, R)], 25: [(0, T14), (5, R)], 26: [(0, R)],
    27: [(0, T02)], 28: [(0, T02)], 29: [(1, T02)], 30: [(0, R), (2, T14)], 31: [(0, R), (1, T14)],
    32: None, 33: [(0, R)], 34: [(0, R)], 35: [(0, T02)], 36: [(0, R)], 37: [(0, R), (1, T14)], 38: [(0, R)],
}
PHOTO_SESSION = "batch_2026-10-08_photos"

# videos: candidates kept when confidence >= KEEP (after these fixes)
KEEP = 0.30
DROP = {  # false positives (PCB, window/monitor, vent, people, blue box, speaker) - (image id, candidate)
    (51, 2), (72, 1), (72, 2), (73, 1), (73, 2), (74, 1), (74, 2), (75, 1), (75, 2), (76, 3), (77, 1),
    (106, 1), (107, 1), (108, 1), (109, 1), (123, 0), (124, 0), (125, 1), (193, 1), (194, 1),
    (199, 2), (200, 2), (243, 3), (247, 2), (283, 0), (295, 1), (300, 1), (301, 1),
}
FORCE = {  # (image id, candidate) -> class, kept whatever the confidence (low-confidence real objects / wrong class)
    (48, 2): T02, (49, 1): T02, (56, 1): T02, (75, 4): R, (120, 0): R, (122, 0): R, (123, 2): R, (124, 1): R,
    (207, 1): T14,
    (256, 1): T14, (258, 1): T14, (259, 0): T14, (260, 0): T14, (261, 0): T14, (262, 0): T14, (263, 0): T14,
    (264, 0): T02, (220, 0): T02, (266, 0): T02, (221, 1): T02, (265, 0): T02, (265, 1): T02,
}
UNION = {71: (1, 2)}   # one instrument split into two boxes -> merge
TEST_EVERY = 4         # v00: every 4th frame -> test, others left out
# v00 test frames checked by hand (exact boxes; the leg false-positive in 000324 removed, missed scopes added)
TEST_FIX = {
    "v00_000038": [(R, 6, 340, 266, 510), (T14, 314, 404, 392, 495)],
    "v00_000073": [(R, 0, 347, 251, 508), (T14, 297, 407, 392, 509)],
    "v00_000285": [(R, 65, 353, 163, 397), (T14, 196, 373, 267, 419)],
    "v00_000324": [(R, 35, 326, 132, 361), (T14, 159, 336, 238, 370)],
}


def teal_vs_beige(img, box):
    hsv = cv2.cvtColor(img[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    teal = ((hsv[:, 0] >= 80) & (hsv[:, 0] <= 105) & (hsv[:, 1] > 70) & (hsv[:, 2] > 50)).mean()
    beige = ((hsv[:, 1] < 60) & (hsv[:, 2] > 120)).mean()
    return teal - beige


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def dedupe(boxes):
    """Same instrument boxed twice: keep the higher-confidence box."""
    boxes = sorted(boxes, key=lambda b: -b[2])
    kept = []
    for cls, box, conf in boxes:
        area = (box[2] - box[0]) * (box[3] - box[1])
        dup = False
        for c2, b2, _ in kept:
            ix = max(0, min(box[2], b2[2]) - max(box[0], b2[0]))
            iy = max(0, min(box[3], b2[3]) - max(box[1], b2[1]))
            if c2 == cls and (iou(box, b2) > 0.5 or ix * iy > 0.8 * area):
                dup = True
        if not dup:
            kept.append((cls, box, conf))
    return kept


def video_boxes(it, img):
    k, cands, vid = it["id"], it["candidates"], it["file"].split("frames_")[1][:3]
    out = []
    for j, c in enumerate(cands):
        if (k, j) in DROP:
            continue
        if (k, j) in FORCE:
            out.append((FORCE[(k, j)], c["box"], max(c["conf"], 0.5)))
            continue
        if c["conf"] < KEEP:
            continue
        cls = c["cls"]
        if cls.startswith("tek"):
            if vid == "v02":
                cls = T02
            elif vid == "v00":
                cls = T14
            else:   # v01: teal-dominant box = Tek lying on its back (TDS 2014); beige front = TDS 1002
                s = teal_vs_beige(img, c["box"])
                if s > 0.4:
                    continue        # teal shirt etc.
                cls = T14 if s > -0.4 else T02
        out.append((cls, c["box"], c["conf"]))
    if k in UNION:
        a, b = (cands[i]["box"] for i in UNION[k])
        merged = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
        out = [o for o in out if o[1] not in (a, b)] + [(R, merged, 0.9)]
    return dedupe(out)


def main():
    items = json.load(open(os.path.join(ROOT, "raw", "review", "candidates.json")))
    path = os.path.join(ROOT, "raw", "real_labels.json")
    labels = json.load(open(path))["images"]
    labels = [l for l in labels if "batch_2026-10-08" not in l["file"]]   # re-runnable
    new, v00 = [], 0
    for it in items:
        f = it["file"]
        if "/photos/" in f:
            dec = PHOTOS.get(it["id"])
            if dec is None:
                continue
            boxes = []
            for d in dec:
                if isinstance(d[0], int):
                    x0, y0, x1, y1 = it["candidates"][d[0]]["box"]
                    cls = d[1]
                else:
                    cls, x0, y0, x1, y1 = d
                boxes.append({"cls": cls, "bbox": [x0, y0, x1 - x0, y1 - y0]})
            new.append({"file": f, "session": PHOTO_SESSION, "view": "", "split": "train", "boxes": boxes})
            continue
        vid = f.split("frames_")[1][:3]
        split = "train"
        if vid == "v00":
            v00 += 1
            if v00 % TEST_EVERY:
                continue
            split = "test"
        img = cv2.imread(os.path.join(ROOT, f))
        boxes = [{"cls": c, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]} for c, b, _ in video_boxes(it, img)]
        key = os.path.splitext(os.path.basename(f))[0]
        if key in TEST_FIX:
            boxes = [{"cls": c, "bbox": [x0, y0, x1 - x0, y1 - y0]} for c, x0, y0, x1, y1 in TEST_FIX[key]]
        new.append({"file": f, "session": f"batch_2026-10-08_{vid}", "view": "", "split": split, "boxes": boxes})

    shutil.copy(path, os.path.join(ROOT, "raw", "real_labels.before_batch.json"))
    with open(path, "w") as fh:
        json.dump({"images": labels + new}, fh, indent=1)
    cnt = {c: sum(1 for n in new for b in n["boxes"] if b["cls"] == c) for c in (R, T14, T02)}
    print(f"added {len(new)} images ({sum(n['split'] == 'test' for n in new)} test), boxes {cnt}")

    # check sheets with the final boxes
    col = {R: (127, 194, 24), T14: (31, 138, 255), T02: (209, 75, 209)}
    tiles = []
    for n in new:
        img = cv2.imread(os.path.join(ROOT, n["file"]))
        for b in n["boxes"]:
            x, y, w, h = b["bbox"]
            cv2.rectangle(img, (x, y), (x + w, y + h), col[b["cls"]], max(3, img.shape[1] // 250))
        s = 220 / max(img.shape[:2])
        t = np.full((232, 220, 3), 255, np.uint8)
        r = cv2.resize(img, None, fx=s, fy=s)
        t[:r.shape[0], :r.shape[1]] = r
        cv2.putText(t, os.path.basename(os.path.dirname(n["file"]))[-6:] + "/" + os.path.basename(n["file"])[-10:-4]
                    + (" T" if n["split"] == "test" else ""), (2, 229), 0, 0.35, (0, 0, 0), 1)
        tiles.append(t)
    os.makedirs(os.path.join(ROOT, "raw", "review"), exist_ok=True)
    for s in range(0, len(tiles), 48):
        ch = tiles[s:s + 48]
        while len(ch) % 8:
            ch.append(np.full((232, 220, 3), 255, np.uint8))
        cv2.imwrite(os.path.join(ROOT, "raw", "review", f"final_{s // 48:02d}.jpg"),
                    np.vstack([np.hstack(ch[i:i + 8]) for i in range(0, len(ch), 8)]))


if __name__ == "__main__":
    main()
