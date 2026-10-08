"""Crops of every Tektronix box for the second-stage classifier (TDS 2014 vs TDS 1002).

Sources (the real test photos are never used):
  - real labelled photos/frames with split "train" (raw/real_labels.json) - full resolution, the best crops
  - datasets/oscilloscopes3 train2017 / val2017 (generated scenes, real cut-outs, renders, crops)
Real crops from a few sessions go to "val" so the classifier is chosen on real images it has not seen.

    .venv/Scripts/python tools/build_tek_crops.py      -> datasets/tek_crops/{train,val}/{tek_tds2014,tek_tds1002}/
"""
import json
import os
import random
import shutil

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "datasets", "oscilloscopes3")
OUT = os.path.join(ROOT, "datasets", "tek_crops")
TEK = ("tek_tds2014", "tek_tds1002")
PAD = 0.10        # same padding the server uses
MIN_SIDE = 24     # smaller boxes carry no readable detail
VAL_SESSIONS = {"batch_2026-10-08_photos", "white_table"}   # real sessions held out for choosing the model


def crop(img, x, y, w, h):
    H, W = img.shape[:2]
    px, py = w * PAD, h * PAD
    x0, y0 = max(0, int(x - px)), max(0, int(y - py))
    x1, y1 = min(W, int(x + w + px)), min(H, int(y + h + py))
    c = img[y0:y1, x0:x1]
    s = 256 / max(c.shape[:2])
    if s < 1:
        c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return c


def main():
    random.seed(0)
    shutil.rmtree(OUT, ignore_errors=True)
    for sp in ("train", "val"):
        for c in TEK:
            os.makedirs(os.path.join(OUT, sp, c), exist_ok=True)
    count = {}

    def save(sp, cls, im, name):
        cv2.imwrite(os.path.join(OUT, sp, cls, name), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        count[(sp, cls)] = count.get((sp, cls), 0) + 1

    for it in json.load(open(os.path.join(ROOT, "raw", "real_labels.json")))["images"]:
        if it.get("split") == "test":
            continue
        boxes = [b for b in it["boxes"] if b["cls"] in TEK and min(b["bbox"][2:]) >= MIN_SIDE]
        if not boxes:
            continue
        img = cv2.imread(os.path.join(ROOT, it["file"]))
        sp = "val" if it.get("session") in VAL_SESSIONS else "train"
        stem = it["file"].replace("/", "_").rsplit(".", 1)[0]
        for k, b in enumerate(boxes):
            save(sp, b["cls"], crop(img, *b["bbox"]), f"real_{stem}_{k}.jpg")

    for split in ("train", "val"):
        coco = json.load(open(os.path.join(DS, "annotations", f"instances_{split}2017.json")))
        names = {c["id"]: c["name"] for c in coco["categories"]}
        imgs = {i["id"]: i["file_name"] for i in coco["images"]}
        by_img = {}
        for a in coco["annotations"]:
            if names[a["category_id"]] in TEK and min(a["bbox"][2:]) >= MIN_SIDE:
                by_img.setdefault(a["image_id"], []).append(a)
        for iid, anns in by_img.items():
            img = cv2.imread(os.path.join(DS, f"{split}2017", imgs[iid]))
            for k, a in enumerate(anns):
                # generated data: only for training (validation must be real images)
                save("train", names[a["category_id"]], crop(img, *a["bbox"]), f"gen_{split}_{iid}_{k}.jpg")

    for k in sorted(count):
        print(f"{k[0]:<5} {k[1]:<12} {count[k]}")


if __name__ == "__main__":
    main()
