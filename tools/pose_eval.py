"""Score the pose stage on an RGB-D sequence in FoundationPose's demo layout (with annotated_poses/).

    .venv/Scripts/python tools/pose_eval.py datasets/pose_test/tek_tds1002 [--backend geometric|foundationpose]
                                            [--detector] [--overlay]

Frames go through pose.service.PoseService exactly like server requests (register, then track, re-register when
lost). The "detection" is the box of the ground-truth mask by default (to test the pose stage alone);
--detector runs the real recognition model instead (the full Recognition -> Pose chain).
Reports per frame and in total: translation error (cm), rotation error (deg), ADD (mean distance of the
8 box corners) and ADD < 10% of the object diameter (the usual pose accuracy measure).
--overlay writes <scene>/eval_<backend>/ with the posed 3D box drawn on every frame.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import registry  # noqa: E402
from pose import geometry as G, meshes  # noqa: E402
from pose.service import PoseService  # noqa: E402

EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_pose(img, K, T, size, colour=(0, 255, 0)):
    uv = G.project(K, G.transform(T, G.box_corners(size))).astype(int)
    for a, b in EDGES:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), colour, 2, cv2.LINE_AA)
    o = G.project(K, G.transform(T, np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])))
    for k, c in ((1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 0, 0))):     # x red, y green, z blue
        cv2.line(img, tuple(o[0].astype(int)), tuple(o[k].astype(int)), c, 2, cv2.LINE_AA)
    return img


def rot_err(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--backend", default="geometric")
    ap.add_argument("--label", default=None, help="instrument (default: the mesh name in <scene>/mesh)")
    ap.add_argument("--detector", action="store_true", help="use the real recognition model for the boxes")
    ap.add_argument("--no-depth", action="store_true", help="pretend there is no depth (like the Quest today)")
    ap.add_argument("--overlay", action="store_true")
    args = ap.parse_args()

    label = args.label or os.path.splitext(os.path.basename(glob.glob(os.path.join(args.scene, "mesh", "*.obj"))[0]))[0]
    insts = registry.load()
    size = meshes.size_m({it["label"]: it for it in insts}[label])
    diameter = float(np.linalg.norm(size))
    svc = PoseService(insts, backend=args.backend, mask_method="grabcut", default_label=label)
    K = np.loadtxt(os.path.join(args.scene, "cam_K.txt")).reshape(3, 3)
    if args.detector:
        os.environ.setdefault("CONTEXT_MODEL", "off")
        import server.app as A

    out_dir = os.path.join(args.scene, f"eval_{args.backend}")
    if args.overlay:
        os.makedirs(out_dir, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(args.scene, "rgb", "*.png"))):
        name = os.path.splitext(os.path.basename(f))[0]
        bgr = cv2.imread(f)
        depth = None if args.no_depth else cv2.imread(f.replace("rgb", "depth"), -1).astype(np.float32) / 1000
        gt = np.loadtxt(os.path.join(args.scene, "annotated_poses", name + ".txt")).reshape(4, 4)
        H, W = bgr.shape[:2]
        if args.detector:
            dets = [d for d in A.TARGET.detect(bgr, A.TARGET_MIN_CONF, True) if d["confidence"] >= A.TARGET_MIN_CONF]
            A.apply_tek_check(bgr, dets)
            A.apply_detector_confirm(dets)
            for d in dets:
                d["target"] = True
        else:
            m = cv2.imread(f.replace("rgb", "masks"), 0) > 0
            ys, xs = np.nonzero(m)
            dets = [{"label": label, "confidence": 1.0, "target": True, "model_certain": True,
                     "bbox": {"x": xs.min() / W, "y": ys.min() / H, "width": (xs.max() - xs.min()) / W,
                              "height": (ys.max() - ys.min()) / H}}]
        r = svc.process(bgr, depth, K, dets, session_id="eval", label=label)
        if "ob_in_cam" not in r:
            rows.append((name, r["state"], None, None, None))
            continue
        T = np.array(r["ob_in_cam"]).reshape(4, 4)
        te = float(np.linalg.norm(T[:3, 3] - gt[:3, 3])) * 100
        re = rot_err(T[:3, :3], gt[:3, :3])
        corners = G.box_corners(size)
        add = float(np.linalg.norm(G.transform(T, corners) - G.transform(gt, corners), axis=1).mean())
        rows.append((name, r["state"], te, re, add))
        if args.overlay:
            vis = draw_pose(bgr.copy(), K, gt, size, (200, 200, 200))
            vis = draw_pose(vis, K, T, size, (0, 255, 0))
            cv2.putText(vis, f"{r['state']}  t {te:.1f} cm  R {re:.1f} deg", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.imwrite(os.path.join(out_dir, name + ".jpg"), vis)

    ok = [r for r in rows if r[2] is not None]
    print(f"{'frame':8} {'state':11} {'t err cm':>8} {'R err deg':>9} {'ADD cm':>7}")
    for name, st, te, re, add in rows[:: max(1, len(rows) // 12)]:
        print(f"{name:8} {st:11} " + (f"{te:8.1f} {re:9.1f} {add * 100:7.1f}" if te is not None else "       -"))
    if ok:
        te = np.array([r[2] for r in ok]); re = np.array([r[3] for r in ok]); add = np.array([r[4] for r in ok])
        print(f"\nbackend {args.backend}{' (no depth)' if args.no_depth else ''}: {len(ok)}/{len(rows)} frames posed")
        print(f"translation error  median {np.median(te):.1f} cm   mean {te.mean():.1f} cm")
        print(f"rotation error     median {np.median(re):.1f} deg  mean {re.mean():.1f} deg")
        print(f"ADD < 10% of diameter ({diameter * 10:.1f} cm): {np.mean(add < 0.1 * diameter):.0%} of frames")
    if args.overlay:
        print("overlays:", os.path.relpath(out_dir, ROOT))


if __name__ == "__main__":
    main()
