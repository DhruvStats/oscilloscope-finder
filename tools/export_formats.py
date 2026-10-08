"""Export the COCO dataset (datasets/oscilloscopes3) to YOLO and Pascal VOC formats as well.

    .venv/Scripts/python tools/export_formats.py

Writes, next to the COCO files:
  yolo/   images/{train,val,test}/  labels/{train,val,test}/*.txt  (class cx cy w h, normalised)  data.yaml
  voc/    JPEGImages/  Annotations/*.xml  ImageSets/Main/{train,val,test}.txt
Images are hard-linked when possible (no extra disk space), otherwise copied.
"""
import json
import os
import shutil
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "datasets", "oscilloscopes3")


def link(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    for d in ("yolo", "voc"):
        shutil.rmtree(os.path.join(DS, d), ignore_errors=True)
    names = None
    counts = {}
    for split in ("train", "val", "test"):
        ann = os.path.join(DS, "annotations", f"instances_{split}2017.json")
        if not os.path.exists(ann):
            continue
        coco = json.load(open(ann))
        cats = sorted(coco["categories"], key=lambda c: c["id"])
        names = [c["name"] for c in cats]
        idx = {c["id"]: i for i, c in enumerate(cats)}
        by_img = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        yi, yl = os.path.join(DS, "yolo", "images", split), os.path.join(DS, "yolo", "labels", split)
        vj, va = os.path.join(DS, "voc", "JPEGImages"), os.path.join(DS, "voc", "Annotations")
        vs = os.path.join(DS, "voc", "ImageSets", "Main")
        for d in (yi, yl, vj, va, vs):
            os.makedirs(d, exist_ok=True)
        ids = []
        for im in coco["images"]:
            src = os.path.join(DS, f"{split}2017", im["file_name"])
            stem = f"{split}_{os.path.splitext(im['file_name'])[0]}"
            W, H = im["width"], im["height"]
            link(src, os.path.join(yi, stem + ".jpg"))
            link(src, os.path.join(vj, stem + ".jpg"))
            lines, objs = [], []
            for a in by_img.get(im["id"], []):
                x, y, w, h = a["bbox"]
                c = idx[a["category_id"]]
                lines.append(f"{c} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
                objs.append(f"""  <object><name>{escape(names[c])}</name><pose>Unspecified</pose><truncated>0</truncated>
    <difficult>0</difficult><bndbox><xmin>{int(x) + 1}</xmin><ymin>{int(y) + 1}</ymin>
    <xmax>{int(x + w)}</xmax><ymax>{int(y + h)}</ymax></bndbox></object>""")
            with open(os.path.join(yl, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            with open(os.path.join(va, stem + ".xml"), "w", encoding="utf-8") as f:
                f.write(f"""<annotation><folder>JPEGImages</folder><filename>{stem}.jpg</filename>
  <size><width>{W}</width><height>{H}</height><depth>3</depth></size><segmented>0</segmented>
{chr(10).join(objs)}
</annotation>
""")
            ids.append(stem)
        with open(os.path.join(vs, f"{split}.txt"), "w") as f:
            f.write("\n".join(ids) + "\n")
        counts[split] = len(ids)
    with open(os.path.join(DS, "yolo", "data.yaml"), "w") as f:
        f.write(f"path: {os.path.join(DS, 'yolo')}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    print("exported", counts, "classes", names)
    print("  COCO :", os.path.join(DS, "annotations"))
    print("  YOLO :", os.path.join(DS, "yolo"))
    print("  VOC  :", os.path.join(DS, "voc"))


if __name__ == "__main__":
    main()
