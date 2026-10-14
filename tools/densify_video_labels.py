"""More real crops from the training videos: label the frames between two hand-checked frames.

Only a sample of each walk-around video was labelled. Between two labelled frames a < b (at most --max-gap
frames apart) the instruments move smoothly, so for a frame t in between:
  - boxes of the same instrument in a and b are paired (same class, overlapping) and interpolated to t,
  - the detector runs on frame t, and a detection takes the class of the interpolated box it overlaps well.
A detection that matches nothing, or an instrument seen in only one of a / b, is skipped - only boxes both
neighbours agree on are written. The class always comes from the hand-checked labels, never from the model.

Only training sessions are used (the test video v00 is never touched).

    .venv/Scripts/python tools/densify_video_labels.py          -> raw/dense_labels.json + raw/batch_*/dense_v0x/
"""
import argparse
import json
import os
import re
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BATCH = os.path.join(ROOT, "raw", "batch_2026-10-08")


def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax1, bx1) - max(a[0], b[0])), max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def load_labels():
    labels = json.load(open(os.path.join(ROOT, "raw", "real_labels.json")))["images"]
    add_p = os.path.join(ROOT, "raw", "label_additions.json")
    adds = json.load(open(add_p)) if os.path.exists(add_p) else {}
    out = {}
    for it in labels:
        boxes = [(b["cls"], b["bbox"]) for b in it["boxes"]] + [(b["cls"], b["bbox"]) for b in adds.get(it["file"], [])]
        out[it["file"]] = (it.get("split"), boxes)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["v01", "v02"])
    ap.add_argument("--max-gap", type=int, default=45, help="max frames between the two labelled neighbours")
    ap.add_argument("--stride", type=int, default=3, help="label every n-th frame in between")
    ap.add_argument("--min-iou", type=float, default=0.5)
    ap.add_argument("--allow-test", action="store_true", help="also densify a test video (for split-in-time checks)")
    ap.add_argument("--out", default=os.path.join(ROOT, "raw", "dense_labels.json"))
    args = ap.parse_args()

    os.environ.setdefault("CONTEXT_MODEL", "off")
    import server.app as A
    target = A.TARGET
    labels = load_labels()
    dense, n_boxes = {}, {}
    for v in args.videos:
        frames = {}
        for f, (split, boxes) in labels.items():
            m = re.search(rf"frames_{v}/{v}_(\d+)\.jpg$", f)
            if m:
                if split == "test" and not args.allow_test:
                    raise SystemExit(f"{v} contains test frames - not used for densifying")
                frames[int(m.group(1))] = boxes
        keys = sorted(frames)
        cap = cv2.VideoCapture(os.path.join(BATCH, "videos", f"{v}.mp4"))
        out_dir = os.path.join(BATCH, f"dense_{v}")
        os.makedirs(out_dir, exist_ok=True)
        for a, b in zip(keys, keys[1:]):
            if b - a > args.max_gap or not frames[a] or not frames[b]:
                continue
            # pair the same instrument in both neighbours
            pairs = []
            for ca, ba in frames[a]:
                cands = [(iou(ba, bb), bb) for cb, bb in frames[b] if cb == ca]
                if cands:
                    best = max(cands, key=lambda c: c[0])
                    if best[0] >= 0.2:
                        pairs.append((ca, ba, best[1]))
            if not pairs:
                continue
            for t in range(a + args.stride, b - 1, args.stride):
                cap.set(cv2.CAP_PROP_POS_FRAMES, t)
                ok, img = cap.read()
                if not ok:
                    continue
                if cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 15:
                    continue
                h, w = img.shape[:2]
                r = (t - a) / (b - a)
                dets = [[d["bbox"]["x"] * w, d["bbox"]["y"] * h, d["bbox"]["width"] * w, d["bbox"]["height"] * h]
                        for d in target.detect(img, 0.3, False)]
                boxes = []
                for cls, ba, bb in pairs:
                    exp = [ba[k] + (bb[k] - ba[k]) * r for k in range(4)]
                    m = max(dets, key=lambda d: iou(d, exp), default=None)
                    if m is not None and iou(m, exp) >= args.min_iou:
                        boxes.append({"cls": cls, "bbox": [int(round(x)) for x in m]})
                if not boxes:
                    continue
                name = f"{v}_{t:06d}.jpg"
                cv2.imwrite(os.path.join(out_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                dense[os.path.relpath(os.path.join(out_dir, name), ROOT).replace("\\", "/")] = boxes
                for bx in boxes:
                    n_boxes[bx["cls"]] = n_boxes.get(bx["cls"], 0) + 1
        print(v, "frames so far", len(dense), n_boxes, flush=True)
    json.dump(dense, open(args.out, "w"), indent=1)
    print("written", os.path.relpath(args.out, ROOT), len(dense), "frames", n_boxes)


if __name__ == "__main__":
    main()
