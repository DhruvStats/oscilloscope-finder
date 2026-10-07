"""Model-assisted labelling: pre-annotate real photos for Label Studio.

Runs the trained oscilloscope model on every image in a folder and writes
  <out>/labelstudio_tasks.json   tasks with predictions (boxes + class), import into Label Studio
  <out>/labeling_config.xml      the matching labelling interface (three classes)
  <out>/review.jpg               contact sheet with the predicted boxes, for a quick look

In Label Studio: create a project, paste labeling_config.xml under Settings > Labeling Interface,
then Import labelstudio_tasks.json. Predictions appear as editable pre-annotations: fix, accept, export COCO.
The exported COCO goes through tools/import_label_studio_coco.py exactly like the joystick dataset.

    .venv/Scripts/python tools/prelabel.py raw/photos --out raw/prelabels \
        --image-url-prefix "/data/local-files/?d=oscilloscopes/photos/"
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))
os.environ.setdefault("TARGET_MIN_CONF", "0.25")

from app import TARGET  # noqa: E402  (loads the same model the server uses)

COLOURS = {"rs_rtb2004": (0, 200, 90), "tek_tds2014": (0, 140, 255), "tek_tds1002": (210, 75, 210)}


def config_xml(names):
    labels = "\n".join(f'    <Label value="{n}"/>' for n in names)
    return f"""<View>
  <Image name="image" value="$image" zoom="true"/>
  <RectangleLabels name="label" toName="image">
{labels}
  </RectangleLabels>
</View>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-conf", type=float, default=0.25)
    ap.add_argument("--image-url-prefix", default="/data/local-files/?d=",
                    help="prefix Label Studio uses to reach the images; the file name is appended")
    args = ap.parse_args()
    if TARGET is None:
        sys.exit("no trained oscilloscope model found")
    os.makedirs(args.out, exist_ok=True)

    files = sorted(f for f in glob.glob(os.path.join(args.folder, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    tasks, tiles = [], []
    counts = {n: 0 for n in TARGET.names}
    for path in files:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        dets = [d for d in TARGET(img, args.min_conf)]
        result = []
        vis = img.copy()
        for i, d in enumerate(dets):
            b = d["bbox"]
            counts[d["label"]] += 1
            result.append({
                "id": f"p{i}", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                "original_width": w, "original_height": h, "image_rotation": 0, "score": d["confidence"],
                "value": {"x": b["x"] * 100, "y": b["y"] * 100, "width": b["width"] * 100,
                          "height": b["height"] * 100, "rotation": 0, "rectanglelabels": [d["label"]]},
            })
            c = COLOURS.get(d["label"], (0, 200, 90))
            p0 = (int(b["x"] * w), int(b["y"] * h))
            p1 = (int((b["x"] + b["width"]) * w), int((b["y"] + b["height"]) * h))
            lw = max(2, w // 300)
            cv2.rectangle(vis, p0, p1, c, lw)
            cv2.putText(vis, f'{d["label"]} {d["confidence"]:.2f}', (p0[0] + 4, max(20, p0[1] - 8)),
                        0, max(0.6, w / 1600), c, lw, cv2.LINE_AA)
        tasks.append({
            "data": {"image": args.image_url_prefix + os.path.basename(path)},
            "predictions": [{"model_version": TARGET.prefix, "score": max([d["confidence"] for d in dets], default=0),
                             "result": result}],
        })
        s = 240 / max(h, w)
        t = np.full((250, 250, 3), 255, np.uint8)
        r = cv2.resize(vis, (int(w * s), int(h * s)))
        t[:r.shape[0], :r.shape[1]] = r
        cv2.putText(t, os.path.basename(path)[:28], (3, 246), 0, 0.4, (0, 0, 0), 1)
        tiles.append(t)

    with open(os.path.join(args.out, "labelstudio_tasks.json"), "w") as f:
        json.dump(tasks, f, indent=1)
    with open(os.path.join(args.out, "labeling_config.xml"), "w") as f:
        f.write(config_xml(TARGET.names))
    while len(tiles) % 8:
        tiles.append(np.full((250, 250, 3), 255, np.uint8))
    cv2.imwrite(os.path.join(args.out, "review.jpg"),
                np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
    print(f"{len(files)} images, predicted boxes per class: {counts}")


if __name__ == "__main__":
    main()
