"""The pose stage between recognition and Unity: pick the instrument, register once, then track.

Per client session (session_id):
  1. Recognition gives the instrument's label, confidence and box (server/app.py).
  2. The label selects the 3D mesh and real size (config/instruments.yaml -> mesh / *_mm).
  3. First frame (or after loss): mask from the box (pose/masks.py) -> backend.register(...).
  4. Next frames: backend.track(...) when the backend can track (FoundationPose), else register again.
  5. Sanity check: the posed 3D box projected into the image must overlap the detector's box; if not, the
     track is considered lost and the next frame registers again.
The answer is ob_in_cam (OpenCV, 4x4) plus the same pose converted for a Unity prefab built from our mesh.
"""
import os
import threading
import time

import numpy as np

from . import geometry as G
from . import masks, meshes
from .backends import make_backend

LOST_IOU = 0.25            # projected 3D box vs detector box below this -> lost
SESSION_TTL_S = 120        # forget a session after this long without frames


class PoseService:
    def __init__(self, instruments, backend="auto", mask_method="auto", default_label=None):
        self.instruments = {it["label"]: it for it in instruments}
        self.backend = make_backend(backend)
        self.mask_method = mask_method
        self.default_label = default_label
        self.sessions = {}
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def mesh_for(self, label):
        inst = self.instruments[label]
        path = meshes.mesh_path(inst)
        if not os.path.exists(path):
            meshes.build(inst)                       # first use: build the box mesh from the registry
        return path, meshes.size_m(inst)

    def poseable(self, label):
        inst = self.instruments.get(label)
        return bool(inst and inst.get("height_mm") and inst.get("depth_mm"))

    def _choose(self, detections, label):
        cands = [d for d in detections if d.get("target") and self.poseable(d["label"])]
        if label:
            cands = [d for d in cands if d["label"] == label]
        elif self.default_label:
            cands = [d for d in cands if d["label"] == self.default_label] or cands
        if not cands:
            return None
        # prefer a confirmed model name, then the most confident box
        return max(cands, key=lambda d: (bool(d.get("model_certain")), d["confidence"]))

    @staticmethod
    def _box_px(box_norm, shape):
        H, W = shape[:2]
        return np.array([box_norm["x"] * W, box_norm["y"] * H, (box_norm["x"] + box_norm["width"]) * W,
                         (box_norm["y"] + box_norm["height"]) * H])

    def _gc(self):
        now = time.time()
        for k in [k for k, s in self.sessions.items() if now - s["seen"] > SESSION_TTL_S]:
            del self.sessions[k]

    # ------------------------------------------------------------------ main entry
    def process(self, bgr, depth, K, detections, session_id="default", label=None, reset=False):
        t0 = time.perf_counter()
        with self.lock:
            self._gc()
            s = self.sessions.get(session_id)
            det = self._choose(detections, label or (s["label"] if s and not reset else None))
            if s is None or reset or (det is not None and det["label"] != s["label"]):
                s = {"label": det["label"] if det else label, "state": "none", "T": None, "frames": 0}
                self.sessions[session_id] = s
            s["seen"] = time.time()
            if s["label"] is None:
                return {"state": "no_target", "reason": "no instrument with a 3D mesh was recognised"}
            mesh_obj, size = self.mesh_for(s["label"])

            T, score, mask_used, action = None, None, None, None
            if s["state"] in ("registered", "tracking") and self.backend.can_track:
                T, score = self.backend.track(s["label"], bgr, depth, K)
                action = "track"
            if T is not None and det is not None:
                if G.iou(G.projected_box(K, T, size), self._box_px(det["bbox"], bgr.shape)) < LOST_IOU:
                    T, action = None, "lost"
            if T is None:
                if det is None:
                    s["state"] = "lost" if s["state"] != "none" else "no_target"
                    return {"state": s["state"], "label": s["label"],
                            "reason": "instrument not detected in this frame"}
                if self.backend.needs_depth and depth is None:
                    return {"state": "needs_depth", "label": s["label"],
                            "reason": f"the {self.backend.name} backend needs a depth image aligned with the frame"}
                mask, mask_used = masks.make_mask(bgr, det["bbox"], self.mask_method)
                T, score = self.backend.register(s["label"], mesh_obj, size, bgr, depth, K, mask, det["bbox"])
                action = "register"
                if T is None:
                    s["state"] = "lost"
                    return {"state": "lost", "label": s["label"], "reason": "registration failed"}
            s["T"], s["frames"] = T, s["frames"] + 1
            s["state"] = "registered" if action == "register" else "tracking"
            check = None
            if det is not None:
                check = round(G.iou(G.projected_box(K, T, size), self._box_px(det["bbox"], bgr.shape)), 3)
            inst = self.instruments[s["label"]]
            return {
                "state": s["state"],
                "label": s["label"],
                "display_name": inst.get("display_name", s["label"]),
                "model_certain": det.get("model_certain") if det else None,
                "backend": self.backend.name,
                "approximate": self.backend.name == "geometric",
                "ob_in_cam": [round(float(v), 6) for v in T.reshape(-1)],
                "frame": "opencv_camera",
                "unity": G.to_unity(T),
                "score": score,
                "box_agreement": check,
                "mask": mask_used,
                "mesh": os.path.relpath(mesh_obj, meshes.ROOT).replace(os.sep, "/"),
                "size_m": [round(v, 4) for v in size],
                "pose_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
