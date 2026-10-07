"""Turn raw/real_labels_draft.json into the reviewed raw/real_labels.json.

Every decision below was made by looking at the photo (see raw/review_*.jpg):
  - NEGATIVES: no PoC oscilloscope in the photo, kept with zero boxes (teaches what is not a target)
  - EXCLUDE: wide scenes where scopes are tiny/partly hidden and cannot be boxed reliably; an unlabelled
    scope in a training image would teach the model the wrong thing
  - MANUAL: boxes drawn by hand (x0, y0, x1, y1) where the model + GrabCut box was wrong or missing
  - TEST: held-out real photos, never used for textures, backgrounds or cut-outs
"""
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NEGATIVES = {"00", "01", "03"}
EXCLUDE = {"15", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "28"}
MANUAL = {
    "02": [("rs_rtb2004", (150, 30, 1325, 650)), ("tek_tds2014", (1300, 195, 1600, 605))],
    "08": [("tek_tds1002", (0, 108, 651, 1485))],
    "10": [("tek_tds1002", (69, 386, 511, 1314))],
    "11": [("tek_tds1002", (0, 229, 700, 814))],
    "12": [("tek_tds1002", (126, 320, 297, 706))],
    "31": [("tek_tds2014", (200, 5, 1280, 655))],
    "32": [("tek_tds2014", (213, 62, 1227, 710))],
    "41": [("tek_tds1002", (117, 543, 651, 1171))],
    "43": [("tek_tds1002", (20, 486, 543, 1100))],
    "45": [("tek_tds1002", (40, 514, 683, 1257))],
    "46": [("tek_tds1002", (109, 500, 651, 1271))],
    "55": [("tek_tds2014", (328, 99, 1341, 614)), ("rs_rtb2004", (1512, 54, 1600, 454))],
}
# the box is right but the model's class was not: the white-table session is the TDS 2014
RECLASS = {"27": "tek_tds2014"}
TEST = {"02", "49", "61",                  # RTB2004 (front with TDS 2014 edge, hand-held, scene)
        "05", "30", "31", "55",            # TDS 2014 (front, side, back, front)
        "11", "14", "39", "44", "45"}      # TDS 1002 (back, front, front, back with GPIB, side)
COLOURS = {"rs_rtb2004": (0, 200, 90), "tek_tds2014": (0, 140, 255), "tek_tds1002": (210, 75, 210)}


def main():
    with open(os.path.join(ROOT, "raw", "real_labels_draft.json")) as f:
        draft = json.load(f)["images"]
    out, tiles = [], []
    for it in draft:
        photo = os.path.splitext(os.path.basename(it["file"]))[0]
        if photo in EXCLUDE:
            continue
        boxes = [{"cls": b["cls"], "bbox": b["bbox"]} for b in it["boxes"]]
        if photo in NEGATIVES:
            boxes = []
        if photo in RECLASS:
            boxes = [dict(b, cls=RECLASS[photo]) for b in boxes]
        if photo in MANUAL:
            boxes = [{"cls": c, "bbox": [x0, y0, x1 - x0, y1 - y0]} for c, (x0, y0, x1, y1) in MANUAL[photo]]
        out.append({"file": it["file"], "session": it["session"], "view": it["view"],
                    "split": "test" if photo in TEST else "train", "boxes": boxes})

        img = cv2.imread(os.path.join(ROOT, it["file"]))
        h, w = img.shape[:2]
        for b in boxes:
            x, y, bw, bh = b["bbox"]
            cv2.rectangle(img, (x, y), (x + bw, y + bh), COLOURS[b["cls"]], max(4, w // 200))
        s = 260 / max(h, w)
        t = np.full((285, 260, 3), 255, np.uint8)
        r = cv2.resize(img, (int(w * s), int(h * s)))
        t[:r.shape[0], :r.shape[1]] = r
        tag = f"{photo} {'TEST' if photo in TEST else ''} {'NEG' if photo in NEGATIVES else ''} {'MANUAL' if photo in MANUAL else ''}"
        cv2.putText(t, tag, (3, 278), 0, 0.42, (0, 0, 0), 1)
        tiles.append(t)

    with open(os.path.join(ROOT, "raw", "real_labels.json"), "w") as f:
        json.dump({"images": out}, f, indent=1)
    while len(tiles) % 9:
        tiles.append(np.full((285, 260, 3), 255, np.uint8))
    cv2.imwrite(os.path.join(ROOT, "raw", "review_final.jpg"),
                np.vstack([np.hstack(tiles[i:i + 9]) for i in range(0, len(tiles), 9)]))
    n_test = sum(1 for o in out if o["split"] == "test")
    counts = {c: sum(1 for o in out for b in o["boxes"] if b["cls"] == c) for c in COLOURS}
    print(f"{len(out)} photos ({len(out) - n_test} train, {n_test} test), boxes per class {counts}")


if __name__ == "__main__":
    main()
