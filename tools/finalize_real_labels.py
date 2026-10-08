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
EXCLUDE = {"15"}
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
# Wide office / bench scenes, boxed by hand on 2026-10-08. The Tektronix in the office session (17-26) shows
# coloured channel buttons and 5 BNC inputs: it is the TDS 2014 (the model had called it TDS 1002).
# The ARCADIA bench (25, 28) is the TDS 1002. The wall socket in 25/28 and the "mitlab" sign in 23 stay
# unlabelled on purpose: they were false alarms and now act as hard negatives.
MANUAL.update({
    "17": [("rs_rtb2004", (177, 108, 525, 746)), ("tek_tds2014", (186, 696, 437, 1246))],
    "18": [("rs_rtb2004", (255, 185, 570, 765)), ("tek_tds2014", (140, 740, 445, 1320))],
    "19": [("tek_tds2014", (97, 329, 308, 500)), ("rs_rtb2004", (315, 264, 447, 446))],
    "20": [("rs_rtb2004", (466, 398, 698, 545)), ("tek_tds2014", (710, 440, 911, 557))],
    "21": [("rs_rtb2004", (294, 722, 577, 1075)), ("tek_tds2014", (568, 759, 717, 932))],
    "22": [("rs_rtb2004", (193, 794, 625, 1040)), ("tek_tds2014", (643, 896, 717, 1096))],
    "23": [("tek_tds2014", (179, 768, 290, 939)), ("rs_rtb2004", (251, 700, 347, 836))],
    "24": [("rs_rtb2004", (171, 329, 616, 590)), ("tek_tds2014", (651, 407, 996, 603))],
    "25": [("tek_tds1002", (685, 340, 864, 428))],
    "26": [("tek_tds2014", (80, 858, 449, 1080)), ("rs_rtb2004", (0, 787, 59, 1045))],
    "28": [("tek_tds1002", (664, 298, 840, 378))],
})
# 2026-10-08 second review: scopes that were in the photo but had no box (a missing box teaches
# "this is not a scope"). Found by running the model at low confidence and checking every candidate.
# In the croissant-table session only one TDS 1002 exists, so a second Tektronix there is the TDS 2014.
MANUAL.update({
    "09": [("tek_tds1002", (583, 372, 881, 571)), ("rs_rtb2004", (340, 185, 435, 325)),
           ("tek_tds2014", (205, 250, 398, 400))],
    "12": [("tek_tds1002", (126, 320, 297, 706)), ("tek_tds2014", (403, 56, 647, 216)),
           ("rs_rtb2004", (645, 25, 717, 222))],
    "14": [("tek_tds1002", (0, 0, 0, 0)), ("tek_tds2014", (0, 25, 302, 181)), ("rs_rtb2004", (245, 0, 395, 140))],
    "52": [("tek_tds2014", (0, 0, 0, 0)), ("tek_tds1002", (324, 329, 473, 396))],
    "57": [("rs_rtb2004", (0, 0, 0, 0)), ("tek_tds2014", (4, 361, 173, 684))],
    "59": [("rs_rtb2004", (0, 0, 0, 0)), ("tek_tds2014", (4, 555, 112, 861))],
    "62": [("rs_rtb2004", (0, 0, 0, 0)), ("tek_tds2014", (2, 112, 67, 495))],
    "63": [("rs_rtb2004", (0, 0, 0, 0)), ("tek_tds2014", (5, 203, 243, 554))],
})
# (0, 0, 0, 0) = keep the photo's existing box of that class (filled in below)
KEEP = (0, 0, 0, 0)
# the box is right but the model's class was not: the white-table session is the TDS 2014
RECLASS = {"27": "tek_tds2014"}
TEST = {"20", "25",                       # user's screenshots: wide office, far ARCADIA bench
        "02", "49", "61",                  # RTB2004 (front with TDS 2014 edge, hand-held, scene)
        "05", "30", "31", "55",            # TDS 2014 (front, side, back, front)
        "11", "14", "39", "44", "45"}      # TDS 1002 (back, front, front, back with GPIB, side)
# openly licensed photos from the web (see raw/web/provenance.csv and the dataset ATTRIBUTION.md)
EXTRA = [
    {"file": "raw/web/tek_tds1002_85b6e23218.jpg", "session": "web_commons", "view": "front", "split": "train",
     "boxes": [{"cls": "tek_tds1002", "bbox": [40, 470, 2390, 1150]}]},   # CC BY-SA 2.5, Berserkerus
]
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
            new = []
            for c, box in MANUAL[photo]:
                if box == KEEP:     # keep the reviewed box of this class from the draft
                    new += [b for b in boxes if b["cls"] == c][:1]
                else:
                    x0, y0, x1, y1 = box
                    new.append({"cls": c, "bbox": [x0, y0, x1 - x0, y1 - y0]})
            boxes = new
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

    out.extend(EXTRA)
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
