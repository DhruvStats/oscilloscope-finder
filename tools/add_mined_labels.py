"""Apply the review of the hard-negative mining (synth/hard_negatives/index.json), 2026-10-09.

Most "false positives" were real oscilloscopes my labels had missed (edge of the frame, background, close-up
views) - an unlabelled scope teaches the model "this is not a scope". Those boxes become labels in
raw/label_additions.json (merged by synth/gen3.py); Tektronix names come from the second-stage classifier.
The true false positives (poster, keyboard, whiteboard, floor tape, paper ...) are kept in
synth/hard_negatives/neg/ and pasted unlabelled into training scenes.

    .venv/Scripts/python tools/add_mined_labels.py
"""
import json
import os
import shutil
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HN = os.path.join(ROOT, "synth", "hard_negatives")

R = "rs_rtb2004"
TEK = "tek"            # name decided by the Tektronix classifier
SCOPES = {0: TEK, 1: TEK, 10: R, 11: R, 12: TEK, 13: TEK, 14: TEK, 15: TEK, 16: TEK, 17: TEK, 18: TEK, 19: R,
          20: TEK, 21: R, 23: R, 25: R, 26: R, 27: R, 28: R, 29: TEK, 30: TEK, 31: TEK, 32: TEK, 33: TEK, 34: R}
NEGATIVES = {2, 3, 4, 5, 6, 7, 8, 9}     # poster, keyboard, whiteboard, blurred blue item, board edge, floor tape, paper
# 22, 24: slivers at the frame edge - too little visible to label either way (left out)

# Speakers and keyboards the model called an oscilloscope, taken from the batch review candidates
# (raw/review/candidates.json: (image id, candidate index)). The mining above misses the speakers because
# they stand right next to an RTB2004 and overlap its box.
REVIEW_NEGATIVES = [(24, 1), (34, 1), (29, 0), (33, 1), (15, 3), (13, 2),   # black PA speakers
                    (3, 2), (6, 4), (7, 1)]                                # keyboards


def main():
    os.environ.setdefault("CONTEXT_MODEL", "off")
    sys.path.insert(0, os.path.join(ROOT, "server"))
    from app import TEK as tek_classifier

    index = json.load(open(os.path.join(HN, "index.json")))
    adds = {}
    for i, cls in SCOPES.items():
        f = index[i]
        x0, y0, x1, y1 = f["box"]
        if cls == TEK:
            img = cv2.imread(os.path.join(ROOT, f["file"]))
            H, W = img.shape[:2]
            name, p = tek_classifier(img, {"x": x0 / W, "y": y0 / H, "width": (x1 - x0) / W, "height": (y1 - y0) / H})
            if p < 0.6:   # still a real scope: an unlabelled scope hurts more than an uncertain name
                print(f"  #{i}: classifier unsure ({name} {p:.2f}) - added with its best guess")
            cls = name
        adds.setdefault(f["file"], []).append({"cls": cls, "bbox": [x0, y0, x1 - x0, y1 - y0], "source": f"mined#{i}"})
    json.dump(adds, open(os.path.join(ROOT, "raw", "label_additions.json"), "w"), indent=1)

    neg = os.path.join(HN, "neg")
    shutil.rmtree(neg, ignore_errors=True)
    os.makedirs(neg)
    for i in NEGATIVES:
        shutil.copy(os.path.join(HN, index[i]["crop"]), os.path.join(neg, index[i]["crop"]))
    cands = {c["id"]: c for c in json.load(open(os.path.join(ROOT, "raw", "review", "candidates.json")))}
    for k, j in REVIEW_NEGATIVES:
        it = cands[k]
        x0, y0, x1, y1 = it["candidates"][j]["box"]
        crop = cv2.imread(os.path.join(ROOT, it["file"]))[y0:y1, x0:x1]
        cv2.imwrite(os.path.join(neg, f"review_{k}_{j}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    n = sum(len(v) for v in adds.values())
    print(f"{n} missed scopes added in {len(adds)} images -> raw/label_additions.json; "
          f"{len(NEGATIVES) + len(REVIEW_NEGATIVES)} false-positive objects -> synth/hard_negatives/neg/")


if __name__ == "__main__":
    main()
