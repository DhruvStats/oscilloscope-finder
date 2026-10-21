"""Build the metric, textured box mesh of every instrument (pose stage) and a preview sheet.

    .venv/Scripts/python tools/build_meshes.py              -> models/meshes/<label>/<label>.obj/.mtl/.png
    .venv/Scripts/python tools/build_meshes.py tek_tds1002  (one instrument)

Sizes come from config/instruments.yaml (width_mm / height_mm / depth_mm), face photos from synth/textures.
The same .obj can be imported into Unity as the reference object of that model (see docs/POSE_STAGE.md for the
frame convention).
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import registry  # noqa: E402
from pose import geometry as G, meshes  # noqa: E402
from pose.render import render  # noqa: E402


def main():
    want = set(sys.argv[1:])
    tiles = []
    for inst in registry.load():
        if want and inst["label"] not in want:
            continue
        obj = meshes.build(inst)
        size = meshes.size_m(inst)
        print(f"{inst['label']}: {os.path.relpath(obj, ROOT)}  size {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} m")
        # preview: the mesh rendered at three views
        K = G.intrinsics(700, 700, 320, 240)
        row = []
        for yaw, pitch in ((0, 15), (35, 25), (150, 20)):
            T = G.facing_camera(np.array([0, 0, 1.0]), yaw, pitch)
            rgb, _, _ = render(os.path.splitext(obj)[0] + ".png", size, T, K, (480, 640),
                               background=np.full((480, 640, 3), 235, np.uint8))
            row.append(rgb)
        bar = np.full((40, 640 * 3, 3), 255, np.uint8)
        cv2.putText(bar, inst["display_name"], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        tiles.append(np.vstack([bar, np.hstack(row)]))
    if tiles:
        out = os.path.join(ROOT, "models", "meshes", "preview.jpg")
        cv2.imwrite(out, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("preview:", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
