"""Object mask for the first (registration) frame, prompted with the detector's box.

FoundationPose needs a pixel mask of the object on the frame it registers; tracking needs none.
  method "sam2"    - Segment Anything 2 with a box prompt (GPU server; best quality). Needs the `sam2` package
                     and a checkpoint (SAM2_CHECKPOINT, SAM2_CONFIG).
  method "grabcut" - OpenCV GrabCut initialised with the box (CPU, no extra model; good on tables).
  method "box"     - the box itself, slightly shrunk (always works; FoundationPose only uses the mask to pick
                     depth points for the initial translation, so a box is often enough for a box-shaped object).
  method "auto"    - sam2 if available, else grabcut.
"""
import os

import cv2
import numpy as np

_SAM = None


def _box_px(box_norm, shape):
    H, W = shape[:2]
    x, y, w, h = box_norm["x"] * W, box_norm["y"] * H, box_norm["width"] * W, box_norm["height"] * H
    return int(max(0, x)), int(max(0, y)), int(min(W, x + w)), int(min(H, y + h))


def box_mask(shape, box_norm, shrink=0.06):
    x0, y0, x1, y1 = _box_px(box_norm, shape)
    dx, dy = int((x1 - x0) * shrink), int((y1 - y0) * shrink)
    m = np.zeros(shape[:2], bool)
    m[y0 + dy:y1 - dy, x0 + dx:x1 - dx] = True
    return m


def grabcut_mask(bgr, box_norm, iterations=4):
    x0, y0, x1, y1 = _box_px(box_norm, bgr.shape)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return box_mask(bgr.shape, box_norm)
    m = np.zeros(bgr.shape[:2], np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, m, (x0, y0, x1 - x0, y1 - y0), bg, fg, iterations, cv2.GC_INIT_WITH_RECT)
    out = (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
    # GrabCut can lose a uniform panel; fall back to the box when it keeps too little of it
    if out.sum() < 0.3 * (x1 - x0) * (y1 - y0):
        return box_mask(bgr.shape, box_norm)
    return out


def sam2_mask(bgr, box_norm):
    global _SAM
    if _SAM is None:
        from sam2.build_sam import build_sam2                          # noqa: imported only on the GPU server
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _SAM = SAM2ImagePredictor(build_sam2(os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml"),
                                             os.environ["SAM2_CHECKPOINT"]))
    _SAM.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    x0, y0, x1, y1 = _box_px(box_norm, bgr.shape)
    masks, scores, _ = _SAM.predict(box=np.array([x0, y0, x1, y1]), multimask_output=False)
    return masks[0].astype(bool)


def available(method):
    if method in ("box", "grabcut"):
        return True
    if method == "sam2":
        try:
            import sam2  # noqa: F401
            return bool(os.environ.get("SAM2_CHECKPOINT"))
        except ImportError:
            return False
    return True


def make_mask(bgr, box_norm, method="auto"):
    if method == "auto":
        method = "sam2" if available("sam2") else "grabcut"
    if method == "sam2":
        return sam2_mask(bgr, box_norm), "sam2"
    if method == "grabcut":
        return grabcut_mask(bgr, box_norm), "grabcut"
    return box_mask(bgr.shape, box_norm), "box"
