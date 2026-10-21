"""Synthetic RGB-D sequence with known poses, in FoundationPose's demo folder layout.

    .venv/Scripts/python tools/make_pose_testset.py tek_tds1002 --frames 60     -> datasets/pose_test/tek_tds1002/

The instrument's box mesh is rendered over a real lab photo along a smooth camera path (the head of a user
walking around the bench), with a background wall/table depth and sensor-like depth noise. Output:
  rgb/000000.png ...           colour frames
  depth/000000.png ...         16-bit PNG, millimetres (FoundationPose's reader divides by 1000)
  masks/000000.png ...         object mask (FoundationPose needs the first one only)
  cam_K.txt                    3x3 intrinsics
  annotated_poses/000000.txt   ground-truth ob_in_cam 4x4 (metres)
  mesh/<label>.obj/.mtl/.png   the mesh used
On the GPU machine the official demo runs on it unchanged:
  python run_demo.py --mesh_file <scene>/mesh/<label>.obj --test_scene_dir <scene> --debug 2
and tools/pose_eval.py scores any backend against the ground truth.
"""
import argparse
import glob
import os
import random
import shutil
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import registry  # noqa: E402
from pose import geometry as G, meshes  # noqa: E402
from pose.render import render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?", default="tek_tds1002")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fx", type=float, default=600.0)
    ap.add_argument("--noise-mm", type=float, default=3.0, help="depth noise (std), like a consumer RGB-D camera")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    inst = {it["label"]: it for it in registry.load()}[args.label]
    obj = meshes.mesh_path(inst)
    if not os.path.exists(obj):
        meshes.build(inst)
    size = meshes.size_m(inst)
    atlas = cv2.imread(os.path.splitext(obj)[0] + ".png")
    out = args.out or os.path.join(ROOT, "datasets", "pose_test", args.label)
    shutil.rmtree(out, ignore_errors=True)
    for d in ("rgb", "depth", "masks", "annotated_poses", "mesh"):
        os.makedirs(os.path.join(out, d))
    for ext in (".obj", ".mtl", ".png"):
        shutil.copy(os.path.splitext(obj)[0] + ext, os.path.join(out, "mesh"))

    W, H = args.width, args.height
    K = G.intrinsics(args.fx, args.fx, W / 2, H / 2)
    np.savetxt(os.path.join(out, "cam_K.txt"), K, fmt="%.6f")
    # background: a real lab photo; its depth: a wall 2.2-2.8 m away with a gentle slope
    photos = sorted(glob.glob(os.path.join(ROOT, "raw", "batch_2026-10-08", "photos", "*.jpg")))
    bg = cv2.resize(cv2.imread(random.choice(photos)), (W, H), interpolation=cv2.INTER_AREA) if photos else \
        np.full((H, W, 3), 180, np.uint8)
    bg_depth = (2.5 + 0.3 * np.linspace(-1, 1, W)[None, :] + 0.1 * np.linspace(-1, 1, H)[:, None]).astype(np.float32)

    n = args.frames
    for i in range(n):
        a = i / max(1, n - 1)
        yaw = -35 + 70 * a                                   # walk from the left to the right of the bench
        pitch = 20 + 8 * np.sin(2 * np.pi * a)               # head height changes a little
        dist = 0.95 + 0.25 * np.cos(np.pi * a)               # step closer in the middle
        t = np.array([0.04 * np.sin(3 * np.pi * a), 0.02 * np.cos(2 * np.pi * a), dist])
        T = G.facing_camera(t, yaw, pitch)
        rgb, depth, mask = render(atlas, size, T, K, (H, W), background=bg)
        d = np.where(mask, depth, bg_depth)
        d = d + rng.normal(0, args.noise_mm / 1000, d.shape).astype(np.float32)
        name = f"{i:06d}"
        cv2.imwrite(os.path.join(out, "rgb", name + ".png"), rgb)
        cv2.imwrite(os.path.join(out, "depth", name + ".png"), np.clip(d * 1000, 0, 65535).astype(np.uint16))
        cv2.imwrite(os.path.join(out, "masks", name + ".png"), mask.astype(np.uint8) * 255)
        np.savetxt(os.path.join(out, "annotated_poses", name + ".txt"), T, fmt="%.6f")
    print(f"{n} frames of {args.label} -> {os.path.relpath(out, ROOT)} (K fx={args.fx}, {W}x{H})")


if __name__ == "__main__":
    main()
