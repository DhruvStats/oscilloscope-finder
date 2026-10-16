"""One dataset per instrument model, built from datasets/oscilloscopes3 (run after export_formats.py).

    .venv/Scripts/python tools/export_per_model.py      -> datasets/per_model/<model>/

For each model (rs_rtb2004, tek_tds2014, tek_tds1002):
  <split>/images/      every image of that split that shows this model (train / val / test as in the main dataset)
  <split>/labels/      YOLO labels with ONE class (0 = this model); the other two models are NOT labelled here,
                       so a single-model detector trained on it learns "the other scopes are not me"
  <split>/crops/       a padded crop of every instance (for classifiers, quick visual checks and the 3D/pose stage)
  real/<split>/        the original full-resolution real photos and video frames of this model (labels + crops),
                       from raw/real_labels.json, raw/label_additions.json and raw/dense_labels.json - the main
                       dataset mixes resized real photos into train/ under generated names
  data.yaml            YOLO dataset file
  README.md            counts per split and real vs generated images
Images are hard-linked where possible (no extra disk space), copied otherwise.
"""
import json
import os
import shutil

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "datasets", "oscilloscopes3")
OUT = os.path.join(ROOT, "datasets", "per_model")
SPLITS = {"train": "train2017", "val": "val2017", "test": "test2017"}
PAD = 0.10


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_real():
    """Original real photos/frames per model, with the hand-checked labels."""
    items = json.load(open(os.path.join(ROOT, "raw", "real_labels.json")))["images"]
    adds_p, dense_p = (os.path.join(ROOT, "raw", n) for n in ("label_additions.json", "dense_labels.json"))
    adds = json.load(open(adds_p)) if os.path.exists(adds_p) else {}
    dense = json.load(open(dense_p)) if os.path.exists(dense_p) else {}
    items = [dict(it, boxes=it["boxes"] + adds.get(it["file"], [])) for it in items]
    items += [{"file": f, "split": "train", "boxes": bs} for f, bs in dense.items()]
    stats = {}
    for it in items:
        split = "test" if it.get("split") == "test" else "train"
        models = sorted({b["cls"] for b in it["boxes"]})
        if not models:
            continue
        src = os.path.join(ROOT, it["file"])
        img = cv2.imread(src)
        if img is None:
            continue
        H, W = img.shape[:2]
        name = it["file"].replace("raw/", "", 1).replace("/", "_")
        stem = os.path.splitext(name)[0]
        for model in models:
            d = os.path.join(OUT, model, "real", split)
            for sub in ("images", "labels", "crops"):
                os.makedirs(os.path.join(d, sub), exist_ok=True)
            link(src, os.path.join(d, "images", name))
            mine = [b["bbox"] for b in it["boxes"] if b["cls"] == model]
            with open(os.path.join(d, "labels", stem + ".txt"), "w") as f:
                for x, y, w, h in mine:
                    f.write(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
            for k, (x, y, w, h) in enumerate(mine):
                if min(w, h) < 16:
                    continue
                x0, y0 = max(0, int(x - PAD * w)), max(0, int(y - PAD * h))
                x1, y1 = min(W, int(x + w + PAD * w)), min(H, int(y + h + PAD * h))
                cv2.imwrite(os.path.join(d, "crops", f"{stem}_{k}.jpg"), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            s = stats.setdefault(model, {}).setdefault(split, {"images": 0, "real_images": 0, "instances": 0})
            s["images"] += 1
            s["real_images"] += 1
            s["instances"] += len(mine)
    return stats


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    stats = {}
    for split, folder in SPLITS.items():
        coco = json.load(open(os.path.join(DS, "annotations", f"instances_{folder}.json")))
        names = {c["id"]: c["name"] for c in coco["categories"]}
        images = {i["id"]: i for i in coco["images"]}
        by_img = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        for iid, anns in by_img.items():
            im = images[iid]
            src = os.path.join(DS, folder, im["file_name"])
            stem = os.path.splitext(im["file_name"])[0]
            real = im["file_name"].startswith("real")
            for model in sorted({names[a["category_id"]] for a in anns}):
                d = os.path.join(OUT, model, split)
                for sub in ("images", "labels", "crops"):
                    os.makedirs(os.path.join(d, sub), exist_ok=True)
                link(src, os.path.join(d, "images", im["file_name"]))
                W, H = im["width"], im["height"]
                mine = [a for a in anns if names[a["category_id"]] == model]
                with open(os.path.join(d, "labels", stem + ".txt"), "w") as f:
                    for a in mine:
                        x, y, w, h = a["bbox"]
                        f.write(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
                img = None
                for k, a in enumerate(mine):
                    x, y, w, h = a["bbox"]
                    if min(w, h) < 16:
                        continue
                    img = cv2.imread(src) if img is None else img
                    x0, y0 = max(0, int(x - PAD * w)), max(0, int(y - PAD * h))
                    x1, y1 = min(W, int(x + w + PAD * w)), min(H, int(y + h + PAD * h))
                    cv2.imwrite(os.path.join(d, "crops", f"{stem}_{k}.jpg"), img[y0:y1, x0:x1],
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                s = stats.setdefault(model, {}).setdefault(split, {"images": 0, "real_images": 0, "instances": 0})
                s["images"] += 1
                s["real_images"] += real
                s["instances"] += len(mine)
    real = export_real()
    for model, per in stats.items():
        per = dict(per, **{f"real_{k}": v for k, v in real.get(model, {}).items()})
        d = os.path.join(OUT, model)
        with open(os.path.join(d, "data.yaml"), "w") as f:
            f.write(f"path: {d.replace(os.sep, '/')}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                    f"names:\n  0: {model}\n")
        rows = "\n".join(f"| {sp} | {v['images']} | {v['real_images']} | {v['instances']} |" for sp, v in per.items())
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write(f"# {model}\n\nOne-model dataset exported from datasets/oscilloscopes3 by tools/export_per_model.py.\n\n"
                    f"| split | images | real photos/frames | instances |\n|---|---|---|---|\n{rows}\n\n"
                    "train/ mixes the resized real photos in under generated names, so its real count shows 0; real_train / "
                    "real_test are the original real photos and frames. The test split holds only real images never "
                    "used for training. Labels are YOLO format with one "
                    "class (0); crops/ has a padded crop of every instance.\n")
        print(model, per)


if __name__ == "__main__":
    main()
