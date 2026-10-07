"""Convert the COCO datasets listed in configs/oscilloscopes.yaml into one YOLO dataset.

Output (data/yolo):
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt   one line per box: <class> <x_center> <y_center> <width> <height> (0-1)
    classes.txt
    data.yaml                       used by YOLO training

Split:
    The synthetic train2017 + val2017 images are pooled and split 80/20 (train/val).
    The real photos in test2017 are kept apart as the final test set, so no augmented or
    synthetic copy of a test photo is ever seen during training.
"""

import json
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "oscilloscopes.yaml"
OUT = ROOT / "data" / "yolo"


def load_coco(source: Path, split: str):
    """One split of a dataset in COCO layout: images/<split>2017 + annotations/instances_<split>2017.json"""
    return load_coco_file(source / "annotations" / f"instances_{split}2017.json", source / "images" / f"{split}2017")


def load_labelstudio(folder: Path):
    """A Label Studio export in "COCO" format, unzipped: result.json + images/"""
    return load_coco_file(folder / "result.json", folder / "images")


def load_coco_file(ann_file: Path, img_dir: Path):
    """Return [(image_path, [(category_name, x, y, w, h), ...], width, height)]."""
    if not ann_file.exists():
        return []

    coco = json.loads(ann_file.read_text(encoding="utf-8"))
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    boxes = {}
    for a in coco["annotations"]:
        boxes.setdefault(a["image_id"], []).append((cat_names[a["category_id"]], *a["bbox"]))

    items = []
    for img in coco["images"]:
        # Label Studio writes file_name as "images/<name>" or "images\<name>", so match on the base name.
        path = img_dir / Path(img["file_name"].replace("\\", "/")).name
        if not path.exists():
            print(f"  warning: {path.name} listed in {ann_file.name} but missing on disk, skipped")
            continue
        items.append((path, boxes.get(img["id"], []), img["width"], img["height"]))
    return items


def to_yolo_lines(boxes, width, height, class_ids):
    lines = []
    for cat, x, y, w, h in boxes:
        if cat not in class_ids:
            continue
        xc, yc = (x + w / 2) / width, (y + h / 2) / height
        lines.append(f"{class_ids[cat]} {xc:.6f} {yc:.6f} {w / width:.6f} {h / height:.6f}")
    return lines


def write_split(items, split, prefix):
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    for path, lines in items:
        stem = f"{prefix}_{path.stem}"  # prefix keeps names unique across datasets
        shutil.copy2(path, OUT / "images" / split / f"{stem}{path.suffix}")
        (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    classes = cfg["classes"]
    rng = random.Random(cfg["split"]["seed"])
    train_fraction = cfg["split"]["train_fraction"]

    if OUT.exists():
        shutil.rmtree(OUT)

    counts = {"train": 0, "val": 0, "test": 0}
    for class_id, c in enumerate(classes):
        source = Path(c["source"])
        if not source.exists():
            sys.exit(f"Dataset folder not found for {c['name']}: {source}")
        # Accept the dataset's category name, or the class name if that was used as the Label Studio label.
        class_ids = {c["coco_category"]: class_id, c["name"]: class_id}
        print(f"[{class_id}] {c['brand']} {c['model']}  <- {source}")

        def convert(items):
            return [(p, to_yolo_lines(b, w, h, class_ids)) for p, b, w, h in items]

        pool = convert(load_coco(source, "train") + load_coco(source, "val"))
        for folder in c.get("real_photos") or []:
            real = convert(load_labelstudio(Path(folder)))
            if not real:
                sys.exit(f"No labelled images found in {folder} (expected result.json + images/)")
            print(f"    + {len(real)} real labelled photos from {folder}")
            pool += real
        rng.shuffle(pool)
        n_train = round(len(pool) * train_fraction)
        test = convert(load_coco(source, "test"))

        write_split(pool[:n_train], "train", c["name"])
        write_split(pool[n_train:], "val", c["name"])
        write_split(test, "test", c["name"])

        print(f"    train {n_train}  val {len(pool) - n_train}  test(real) {len(test)}"
              f"  | images without a scope: {sum(not l for _, l in pool)}")
        counts["train"] += n_train
        counts["val"] += len(pool) - n_train
        counts["test"] += len(test)

    names = [c["name"] for c in classes]
    (OUT / "classes.txt").write_text("\n".join(names) + "\n")
    data_yaml = {
        "path": str(OUT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(names)),
    }
    (OUT / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"\nTotal: {counts}\nWrote {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
