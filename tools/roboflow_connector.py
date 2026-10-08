"""Optional Roboflow connector - OFF by default, nothing leaves the PC unless explicitly allowed.

Uses Roboflow's official SDK from its own environment (.roboflow-venv), so the main project is untouched.

    .roboflow-venv/Scripts/python tools/roboflow_connector.py status
    .roboflow-venv/Scripts/python tools/roboflow_connector.py push  [--what real|synthetic|all] [--send]
    .roboflow-venv/Scripts/python tools/roboflow_connector.py generate              (dataset version, x3 augmented)
    .roboflow-venv/Scripts/python tools/roboflow_connector.py train --version N     (Roboflow GPU training)
    .roboflow-venv/Scripts/python tools/roboflow_connector.py pull  --version N
    .roboflow-venv/Scripts/python tools/roboflow_connector.py eval  --version N [--send]

push  : upload the dataset (YOLO format, train/val/test splits) to a Roboflow project, to label as a team or
        to train Roboflow's own models there.
pull  : download a dataset version labelled/edited in Roboflow (COCO) into datasets/roboflow_v<N>/; with
        --to-labels the real-photo boxes are merged back into raw/real_labels.json (old file backed up).
eval  : run a model trained in Roboflow on our 23 real test images and print the same score as
        tools/eval_app.py, for a fair comparison with our YOLOX model.

Safety (Leonardo data rules): every command that sends images is a dry run unless ALL of these are true:
  1. ROBOFLOW_API_KEY is set           (your key - set it in the shell, never paste it in a chat or a file)
  2. ROBOFLOW_UPLOAD_APPROVED=yes       (set only after Leonardo approved sending lab photos to Roboflow)
  3. --send is given on the command line
--what: "labelled" = the ~355 real photos and video frames with their checked boxes (raw/real_labels.json),
for review in Roboflow; "all" = the whole generated dataset; "real" = only the 23 real test photos (real training photos are mixed into
the generated set); "synthetic" = train/val only (generated scenes and real crops - they still show the lab).
Settings: ROBOFLOW_WORKSPACE (default: your default workspace), ROBOFLOW_PROJECT (default oscilloscopes-poc).
"""
import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "datasets", "oscilloscopes3")
PROJECT = os.environ.get("ROBOFLOW_PROJECT", "oscilloscopes-poc")


def gate(send):
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    approved = os.environ.get("ROBOFLOW_UPLOAD_APPROVED", "").lower() == "yes"
    problems = []
    if not key:
        problems.append("ROBOFLOW_API_KEY is not set")
    if not approved:
        problems.append("ROBOFLOW_UPLOAD_APPROVED is not 'yes' (needs Leonardo approval for cloud upload)")
    if not send:
        problems.append("--send not given")
    return key, problems


def workspace(key):
    import roboflow
    rf = roboflow.Roboflow(api_key=key)
    ws = os.environ.get("ROBOFLOW_WORKSPACE")
    return rf.workspace(ws) if ws else rf.workspace()


def split_files(what):
    """(split, image, yolo-label) triples from the exported YOLO dataset."""
    out = []
    for split in ("train", "val", "test"):
        idir = os.path.join(DS, "yolo", "images", split)
        if not os.path.isdir(idir):
            continue
        for f in sorted(os.listdir(idir)):
            real = "real_" in f or (split == "test")
            if what == "real" and not real or what == "synthetic" and real or what == "val" and split != "val":
                continue
            stem = os.path.splitext(f)[0]
            out.append((split, os.path.join(idir, f), os.path.join(DS, "yolo", "labels", split, stem + ".txt")))
    return out


def cmd_status(_):
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    print("Roboflow connector")
    print("  API key set          :", bool(key))
    print("  upload approved      :", os.environ.get("ROBOFLOW_UPLOAD_APPROVED", "") == "yes")
    print("  project name         :", PROJECT)
    print("  local dataset (YOLO) :", os.path.isdir(os.path.join(DS, "yolo")))
    if key:
        try:
            ws = workspace(key)
            print("  workspace            :", ws.url, "- projects:", [p.split("/")[-1] for p in ws.projects()])
        except Exception as e:  # wrong key / no network
            print("  workspace            : could not connect:", str(e)[:200])


def labelled_files():
    """Real photos + video frames from raw/real_labels.json as (split, image, temporary YOLO label)."""
    import cv2
    tmp = os.path.join(ROOT, "datasets", "roboflow_push_labels")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    classes = [l.split(":", 1)[1].strip() for l in open(os.path.join(DS, "yolo", "data.yaml"))
               if l.startswith("  ") and ":" in l]
    out = []
    for it in json.load(open(os.path.join(ROOT, "raw", "real_labels.json")))["images"]:
        img = os.path.join(ROOT, it["file"])
        h, w = cv2.imread(img).shape[:2]
        lab = os.path.join(tmp, it["file"].replace("/", "__").rsplit(".", 1)[0] + ".txt")
        with open(lab, "w") as f:
            for b in it["boxes"]:
                x, y, bw, bh = b["bbox"]
                f.write(f"{classes.index(b['cls'])} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                        f"{bw / w:.6f} {bh / h:.6f}\n")
        out.append(("test" if it.get("split") == "test" else "train", img, lab))
    return out


def cmd_push(a):
    files = labelled_files() if a.what == "labelled" else split_files(a.what)
    if a.limit:
        step = max(1, len(files) // a.limit)
        files = files[::step][:a.limit]
    by = {s: sum(1 for x in files if x[0] == s) for s in ("train", "val", "test")}
    print(f"would upload {len(files)} images ({by}) with YOLO boxes to project '{PROJECT}'")
    key, problems = gate(a.send)
    if problems:
        print("DRY RUN - nothing sent:", "; ".join(problems))
        return
    import roboflow  # noqa: F401
    ws = workspace(key)
    names = [p.split("/")[-1] for p in ws.projects()]
    proj = ws.project(PROJECT) if PROJECT in names else ws.create_project(
        PROJECT, "object-detection", "private", "oscilloscope")
    classes = os.path.join(DS, "yolo", "data.yaml")
    batch = "real_labelled" if a.what == "labelled" else "oscilloscopes3"
    ok, failed = 0, []
    for i, (split, img, lab) in enumerate(files, 1):
        rf_split = split if split != "val" else "valid"
        empty = not os.path.exists(lab) or not open(lab).read().strip()
        try:
            if empty:
                # no oscilloscope in this photo: Roboflow rejects an empty annotation, so send the image only
                proj.upload(img, split=rf_split, batch_name=batch, num_retry_uploads=2,
                            tag_names=["no-oscilloscope"])
            else:
                # the SDK reads YOLO .txt labels together with the class names from data.yaml
                proj.upload(img, annotation_path=lab, split=rf_split, batch_name=batch, num_retry_uploads=2,
                            annotation_labelmap=classes)
            ok += 1
        except Exception as e:  # keep going; report at the end (re-running is safe, duplicates are skipped)
            failed.append((os.path.relpath(img, ROOT), str(e)[:120]))
        if i % 25 == 0:
            print(f"  {i}/{len(files)}  ok {ok}  failed {len(failed)}", flush=True)
    print(f"uploaded {ok}/{len(files)} images to {proj.id}")
    for f, e in failed[:20]:
        print("  FAILED", f, "-", e)


def cmd_pull(a):
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        sys.exit("ROBOFLOW_API_KEY is not set")
    ws = workspace(key)
    out = os.path.join(ROOT, "datasets", f"roboflow_v{a.version}")
    ws.project(PROJECT).version(a.version).download("coco", location=out, overwrite=True)
    print("downloaded to", out)
    if a.to_labels:
        merge_labels(out)


def merge_labels(folder):
    """Boxes from a Roboflow COCO export -> raw/real_labels.json, matched by our real image file names."""
    path = os.path.join(ROOT, "raw", "real_labels.json")
    labels = json.load(open(path))["images"]
    by_stem = {os.path.splitext(os.path.basename(it["file"]))[0]: it for it in labels}
    changed = 0
    for split in ("train", "valid", "test"):
        ann = os.path.join(folder, split, "_annotations.coco.json")
        if not os.path.exists(ann):
            continue
        coco = json.load(open(ann))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        for im in coco["images"]:
            # Roboflow renames files "<stem>_jpg.rf.<hash>.jpg"; our stems are "<split>_real_<...>"
            stem = im["file_name"].split("_jpg.rf.")[0]
            stem = stem.split("real_", 1)[-1] if "real_" in stem else None
            it = next((v for k, v in by_stem.items() if stem and stem.endswith(k)), None)
            if it is None:
                continue
            it["boxes"] = [{"cls": cats[x["category_id"]], "bbox": [round(v) for v in x["bbox"]]}
                           for x in coco["annotations"] if x["image_id"] == im["id"]
                           and cats[x["category_id"]] in ("rs_rtb2004", "tek_tds2014", "tek_tds1002")]
            changed += 1
    shutil.copy(path, os.path.join(ROOT, "raw", "real_labels.before_roboflow.json"))
    json.dump({"images": labels}, open(path, "w"), indent=1)
    print(f"merged boxes for {changed} real images into raw/real_labels.json (backup kept)")


def cmd_eval(a):
    tests = [f for f in split_files("all") if f[0] == "test"]
    print(f"would send {len(tests)} real test images to the Roboflow-hosted model {PROJECT}/{a.version}")
    key, problems = gate(a.send)
    if problems:
        print("DRY RUN - nothing sent:", "; ".join(problems))
        return
    ws = workspace(key)
    model = ws.project(PROJECT).version(a.version).model
    if model is None:
        sys.exit(f"version {a.version} has no trained model yet - train it first (train --version {a.version}) and wait "
                 "until the status is 'finished'.")
    coco = json.load(open(os.path.join(DS, "annotations", "instances_test2017.json")))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    right = wrong = missed = false = 0
    for im in coco["images"]:
        pred = model.predict(os.path.join(DS, "test2017", im["file_name"]), confidence=40).json()["predictions"]
        boxes = [([p["x"] - p["width"] / 2, p["y"] - p["height"] / 2, p["x"] + p["width"] / 2,
                   p["y"] + p["height"] / 2], p["class"]) for p in pred]
        used = set()
        for g in [g for g in coco["annotations"] if g["image_id"] == im["id"]]:
            x, y, w, h = g["bbox"]
            gb = [x, y, x + w, y + h]

            def iou(b):
                ix = max(0, min(gb[2], b[2]) - max(gb[0], b[0]))
                iy = max(0, min(gb[3], b[3]) - max(gb[1], b[1]))
                return ix * iy / max(1, w * h + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy)
            best = max(((iou(b[0]), i) for i, b in enumerate(boxes) if i not in used), default=(0, -1))
            if best[0] >= 0.4:
                used.add(best[1])
                right += boxes[best[1]][1] == names[g["category_id"]]
                wrong += boxes[best[1]][1] != names[g["category_id"]]
            else:
                missed += 1
        false += len(boxes) - len(used)
    print(f"Roboflow model: right {right}/{right + wrong + missed}  wrong name {wrong}  missed {missed}  "
          f"false alarms {false}   (compare: tools/eval_app.py for our YOLOX model)")


GEN_SETTINGS = {
    "preprocessing": {"auto-orient": True, "resize": {"width": 640, "height": 640, "format": "Fit within"}},
    # no horizontal flip: mirroring changes the front panels (text, button order) the model must recognise
    "augmentation": {
        "image": {"versions": 3},
        "brightness": {"brighten": True, "darken": True, "percent": 25},
        "exposure": {"percent": 15},
        "blur": {"pixels": 1.5},
        "noise": {"percent": 2},
        "rotate": {"degrees": 10},
    },
}


def cmd_generate(a):
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        sys.exit("ROBOFLOW_API_KEY is not set")
    proj = workspace(key).project(PROJECT)
    print("generating a dataset version (augmentation x3, no flips) - this can take a few minutes ...")
    v = proj.generate_version(GEN_SETTINGS)
    print(f"version {v} created: https://app.roboflow.com/{proj.id}/{v}")


def cmd_train(a):
    """Start a Roboflow training run and wait for it (the SDK's legacy train() crashes without a model type)."""
    import time
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        sys.exit("ROBOFLOW_API_KEY is not set")
    proj = workspace(key).project(PROJECT)
    ver = proj.version(a.version)
    print(f"starting Roboflow training: model {a.model}, version {a.version} (Roboflow GPU, ~20-60 min) ...")
    splits = getattr(ver, "splits", None) or {}
    if splits and not splits.get("valid"):
        sys.exit(f"version {a.version} has no Valid images {splits}. Run:  push --what val --limit 100 --send"
                 "  then  generate  and train the NEW version number it prints.")
    tr = ver.create_training(model_type=a.model, speed=a.speed, epochs=a.epochs)
    print(f"training id {getattr(tr, 'training_id', '?')} - follow it at https://app.roboflow.com/{proj.id}/{a.version}")
    if a.no_wait:
        return
    last = None
    while True:
        status = tr.refresh().status
        if status != last:
            print(time.strftime("%H:%M"), "status:", status, flush=True)
            last = status
        if status in ("finished", "failed", "cancelled", "error"):
            break
        time.sleep(60)
    if status == "finished":
        print("done - compare with our model:  tools/roboflow_connector.py eval --version", a.version, "--send")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("push")
    p.add_argument("--what", choices=["labelled", "real", "synthetic", "val", "all"], default="labelled",
                   help="val = our validation images -> Roboflow 'Valid' split (needed before training)")
    p.add_argument("--limit", type=int, default=0, help="upload at most N images (evenly spread)")
    p.add_argument("--send", action="store_true")
    p = sub.add_parser("pull")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--to-labels", action="store_true")
    sub.add_parser("generate")
    p = sub.add_parser("train")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--model", default="rfdetr-small",
                   help="Roboflow model id, e.g. rfdetr-nano / rfdetr-small / rfdetr-medium (Apache-2.0)")
    p.add_argument("--speed", default=None, help="optional preset, e.g. fast")
    p.add_argument("--no-wait", action="store_true", help="start the run and return immediately")
    p.add_argument("--epochs", type=int, default=None)
    p = sub.add_parser("eval")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--send", action="store_true")
    a = ap.parse_args()
    {"status": cmd_status, "push": cmd_push, "pull": cmd_pull, "eval": cmd_eval,
     "generate": cmd_generate, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
