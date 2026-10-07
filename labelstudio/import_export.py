"""Convert a Label Studio JSON export back into raw/real_labels.json (the file the dataset generator reads).

In Label Studio: Export -> JSON. Then

    .venv/Scripts/python labelstudio/import_export.py path/to/export.json

- every task with a submitted annotation replaces that photo's entry (boxes, status)
- status "exclude ..." drops the photo; "no target oscilloscope" keeps it with zero boxes
- photos never annotated keep their previous entry; split/session come from the task data
The old file is kept as raw/real_labels.backup.json. Afterwards rebuild the dataset and retrain.
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = {"rs_rtb2004", "tek_tds2014", "tek_tds1002"}


def main():
    export = json.load(open(sys.argv[1], encoding="utf-8"))
    path = os.path.join(ROOT, "raw", "real_labels.json")
    old = json.load(open(path))["images"]
    by_file = {it["file"]: it for it in old}
    changed = excluded = unsure = 0
    for task in export:
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        ann = max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        data = task["data"]
        boxes, status = [], ""
        for r in ann["result"]:
            if r["type"] == "rectanglelabels":
                v, W, H = r["value"], r["original_width"], r["original_height"]
                cls = v["rectanglelabels"][0]
                if cls not in CLASSES:
                    continue
                x, y = round(v["x"] * W / 100), round(v["y"] * H / 100)
                w, h = round(v["width"] * W / 100), round(v["height"] * H / 100)
                if w > 3 and h > 3:
                    boxes.append({"cls": cls, "bbox": [x, y, w, h]})
            elif r["type"] == "choices":
                status = r["value"]["choices"][0]
        f = data["file"]
        if status.startswith("exclude"):
            by_file.pop(f, None)
            excluded += 1
            continue
        unsure += status == "unsure which model"
        by_file[f] = {"file": f, "session": data.get("session", ""), "view": data.get("view", ""),
                      "split": data.get("split", "train"), "boxes": boxes, "status": status}
        changed += 1
    shutil.copy(path, os.path.join(ROOT, "raw", "real_labels.backup.json"))
    with open(path, "w") as fh:
        json.dump({"images": list(by_file.values())}, fh, indent=1)
    print(f"updated {changed} photos, excluded {excluded}, marked unsure {unsure}; "
          f"{len(by_file)} photos in raw/real_labels.json")


if __name__ == "__main__":
    main()
