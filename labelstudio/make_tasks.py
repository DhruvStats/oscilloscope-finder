"""Build Label Studio tasks with pre-filled boxes.

  - photos already in raw/real_labels.json: their reviewed boxes (model_version "reviewed")
  - any other photos in the folders given with --new: boxes proposed by the trained model ("model")

Images are referenced through Label Studio's local-files storage, rooted at the project's raw/ folder,
so nothing is copied or uploaded.

    .venv/Scripts/python labelstudio/make_tasks.py [--new raw/new_photos]
writes labelstudio/tasks.json
"""
import argparse
import glob
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")


def url(path):
    rel = os.path.relpath(os.path.join(ROOT, path), RAW).replace(os.sep, "/")
    return "/data/local-files/?d=" + rel


def region(i, cls, x, y, w, h, W, H, score=None):
    r = {"id": f"r{i}", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
         "original_width": W, "original_height": H, "image_rotation": 0,
         "value": {"x": 100 * x / W, "y": 100 * y / H, "width": 100 * w / W, "height": 100 * h / H,
                   "rotation": 0, "rectanglelabels": [cls]}}
    if score is not None:
        r["score"] = score
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", nargs="*", default=[], help="folders under raw/ with photos not labelled yet")
    args = ap.parse_args()
    tasks = []

    labels = json.load(open(os.path.join(RAW, "real_labels.json")))["images"]
    for it in labels:
        if not it["file"].startswith("raw/"):
            continue
        H, W = cv2.imread(os.path.join(ROOT, it["file"])).shape[:2]
        regions = [region(i, b["cls"], *b["bbox"], W, H) for i, b in enumerate(it["boxes"])]
        tasks.append({"data": {"image": url(it["file"]), "file": it["file"], "session": it["session"],
                               "split": it["split"], "view": it.get("view", "")},
                      "predictions": [{"model_version": "reviewed", "result": regions}]})

    if args.new:
        sys.path.insert(0, os.path.join(ROOT, "server"))
        os.environ.setdefault("CONTEXT_MODEL", "off")
        from app import TARGET
        for folder in args.new:
            for p in sorted(glob.glob(os.path.join(ROOT, folder, "*"))):
                if not p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                img = cv2.imread(p)
                H, W = img.shape[:2]
                regions = []
                for i, d in enumerate(TARGET.detect(img, 0.3)):
                    b = d["bbox"]
                    regions.append(region(i, d["label"], b["x"] * W, b["y"] * H, b["width"] * W, b["height"] * H,
                                          W, H, d["confidence"]))
                rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
                tasks.append({"data": {"image": url(rel), "file": rel, "session": os.path.basename(folder),
                                       "split": "train", "view": ""},
                              "predictions": [{"model_version": "model", "result": regions}]})

    out = os.path.join(ROOT, "labelstudio", "tasks.json")
    with open(out, "w") as f:
        json.dump(tasks, f, indent=1)
    print(f"{len(tasks)} tasks -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
