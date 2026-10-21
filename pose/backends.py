"""Pose estimators behind one interface: register (first frame, needs a mask) and track (next frames).

  FoundationPoseBackend  the official NVlabs implementation (GPU + CUDA, RGB-D). Uses exactly the calls of its
                         run_demo.py: FoundationPose(...), est.register(K, rgb, depth, ob_mask, iteration),
                         est.track_one(rgb, depth, K, iteration). Set FOUNDATIONPOSE_DIR to the cloned repo
                         (with weights/ downloaded), see docs/POSE_STAGE.md.
  GeometricBackend       CPU fallback: translation from depth (or, without depth, from the known instrument
                         size and the box width), orientation "upright, front facing the camera". Approximate -
                         for testing the Recognition -> Pose -> Unity chain end to end, not for accurate AR.

All poses are ob_in_cam 4x4 matrices in the OpenCV camera frame (pose/geometry.py).
"""
import os
import sys

import numpy as np

from . import geometry as G


class PoseBackend:
    name = "base"
    needs_depth = False
    can_track = False

    def register(self, label, mesh_obj, size_m, bgr, depth, K, mask, box_norm):
        raise NotImplementedError

    def track(self, label, bgr, depth, K):
        raise NotImplementedError


class GeometricBackend(PoseBackend):
    name = "geometric"

    def register(self, label, mesh_obj, size_m, bgr, depth, K, mask, box_norm):
        H, W = bgr.shape[:2]
        width, height, dep = size_m
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None, None
        u, v = xs.mean(), ys.mean()
        z = None
        if depth is not None:
            d = depth[mask]
            d = d[d > 0.05]
            if len(d) > 20:
                z = float(np.median(d)) + dep / 2          # visible surface ~ front panel -> box centre
        if z is None:
            # pinhole: an instrument of known width that spans w pixels is at fx * width / w
            w_px = box_norm["width"] * W
            z = float(K[0, 0] * width / max(w_px, 1)) + dep / 2
        t = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        return G.facing_camera(t), None

    def track(self, label, bgr, depth, K):
        return None, None          # no temporal model: the service re-registers from the next detection


class FoundationPoseBackend(PoseBackend):
    name = "foundationpose"
    needs_depth = True
    can_track = True

    def __init__(self, fp_dir=None, est_refine_iter=5, track_refine_iter=2, debug=0):
        self.fp_dir = fp_dir or os.environ.get("FOUNDATIONPOSE_DIR")
        if not self.fp_dir or not os.path.isdir(self.fp_dir):
            raise RuntimeError("FOUNDATIONPOSE_DIR must point to the cloned NVlabs/FoundationPose repository")
        sys.path.insert(0, self.fp_dir)
        # the official modules (built inside its Docker image / conda env with nvdiffrast, pytorch3d, mycpp)
        import estimater  # noqa: F401
        import nvdiffrast.torch as dr
        import trimesh
        self._fp, self._trimesh = estimater, trimesh
        self.scorer = estimater.ScorePredictor()
        self.refiner = estimater.PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est_iter, self.track_iter, self.debug = est_refine_iter, track_refine_iter, debug
        self.debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models",
                                      "pose_debug")
        self.estimators = {}

    def _estimator(self, label, mesh_obj):
        if label not in self.estimators:
            mesh = self._trimesh.load(mesh_obj, force="mesh")
            self.estimators[label] = self._fp.FoundationPose(
                model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=self.scorer,
                refiner=self.refiner, debug_dir=self.debug_dir, debug=self.debug, glctx=self.glctx)
        return self.estimators[label]

    def register(self, label, mesh_obj, size_m, bgr, depth, K, mask, box_norm):
        if depth is None:
            raise ValueError("FoundationPose needs a depth image aligned with the colour image")
        est = self._estimator(label, mesh_obj)
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        pose = est.register(K=K, rgb=rgb, depth=depth.astype(np.float32), ob_mask=mask.astype(bool),
                            iteration=self.est_iter)
        scores = getattr(est, "scores", None)
        score = float(scores[0]) if scores is not None and len(scores) else None
        return np.asarray(pose, float), score

    def track(self, label, bgr, depth, K):
        est = self.estimators.get(label)
        if est is None or depth is None:
            return None, None
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        pose = est.track_one(rgb=rgb, depth=depth.astype(np.float32), K=K, iteration=self.track_iter)
        return np.asarray(pose, float), None


def make_backend(name="auto"):
    """'foundationpose', 'geometric' or 'auto' (FoundationPose when FOUNDATIONPOSE_DIR and CUDA are there)."""
    if name == "auto":
        try:
            import torch
            cuda = torch.cuda.is_available()
        except ImportError:
            cuda = False
        name = "foundationpose" if cuda and os.environ.get("FOUNDATIONPOSE_DIR") else "geometric"
    if name == "foundationpose":
        return FoundationPoseBackend(est_refine_iter=int(os.environ.get("FP_EST_REFINE_ITER", "5")),
                                     track_refine_iter=int(os.environ.get("FP_TRACK_REFINE_ITER", "2")))
    if name == "geometric":
        return GeometricBackend()
    raise ValueError(f"unknown pose backend {name!r}")
