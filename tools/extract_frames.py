"""Turn a walk-around video into a set of sharp, non-duplicate photos for the dataset.

    .venv/Scripts/python tools/extract_frames.py video.mp4 raw/frames/tds2014_session1 --count 220

Frames are sampled evenly, blurry ones (low Laplacian variance) are dropped, and near-duplicates
(small difference to the last kept frame) are skipped, so the result covers the whole walk-around.
Name the output folder after the instrument and session: the session becomes the split unit.
"""
import argparse
import os

import cv2
import numpy as np


def sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--count", type=int, default=220, help="target number of frames")
    ap.add_argument("--max-side", type=int, default=1280)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise SystemExit(f"cannot read {args.video}")
    # look at 3x more candidates than needed, keep the sharpest of each small window
    candidates = np.linspace(0, total - 1, args.count * 3).astype(int)
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]

    kept, last_small, window = 0, None, []
    for i, idx in enumerate(candidates):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        window.append((sharpness(gray), idx, frame))
        if len(window) < 3 and i < len(candidates) - 1:
            continue
        score, idx, frame = max(window, key=lambda w: w[0])
        window = []
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
        if score < 40:                                   # motion blur / out of focus
            continue
        if last_small is not None and np.abs(small - last_small).mean() < 4:   # camera barely moved
            continue
        last_small = small
        s = args.max_side / max(frame.shape[:2])
        if s < 1:
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(args.out, f"{stem}_{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        kept += 1
    print(f"{kept} frames written to {args.out} (from {total} video frames)")


if __name__ == "__main__":
    main()
