"""CPU renderer for the box meshes: colour, metric depth and mask at a known pose (synthetic RGB-D with ground truth).

Used to test the pose stage end to end without a depth camera, and to write FoundationPose demo folders whose
correct answer is known (tools/make_pose_testset.py). Faces are drawn as textured planes with a z-buffer.
"""
import numpy as np
import cv2

from . import geometry as G
from .meshes import ORDER, FACES, CELL


def render(mesh_png, size_m, T, K, shape, background=None):
    """Return (rgb uint8 HxWx3 BGR, depth float32 metres (0 = no object), mask bool) of the posed box."""
    H, W = shape
    atlas = cv2.imread(mesh_png, cv2.IMREAD_COLOR) if isinstance(mesh_png, str) else mesh_png
    rgb = np.zeros((H, W, 3), np.uint8) if background is None else background.copy()
    zbuf = np.full((H, W), np.inf, np.float32)
    hw, hh, hd = (s / 2 for s in size_m)
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5, np.arange(H, dtype=np.float32) + 0.5)
    rays = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
    for k, face in enumerate(ORDER):
        corners = np.array([(sx * hw, sy * hh, sz * hd) for sx, sy, sz in FACES[face]])
        pc = G.transform(T, corners)
        if (pc[:, 2] <= 0.01).any():
            continue
        # (top-right - top-left) x (bottom-left - top-left) points INTO the box; the face is visible when the
        # camera (origin) is on the outside, i.e. that inward normal points away from the camera
        normal = np.cross(pc[1] - pc[0], pc[3] - pc[0])
        if normal @ pc[0] <= 0:
            continue
        uv = G.project(K, pc).astype(np.float32)
        r, c = divmod(k, 3)
        cell = atlas[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
        M = cv2.getPerspectiveTransform(np.float32([(0, 0), (CELL, 0), (CELL, CELL), (0, CELL)]), uv)
        warped = cv2.warpPerspective(cell, M, (W, H), flags=cv2.INTER_LINEAR)
        inside = cv2.fillConvexPoly(np.zeros((H, W), np.uint8), np.round(uv).astype(np.int32), 1).astype(bool)
        # depth of the face plane along each pixel ray: n.(s*ray) = n.p0
        s = (normal @ pc[0]) / (rays @ normal)
        z = (s * rays[..., 2]).astype(np.float32)
        shade = 0.75 + 0.25 * abs(normal[2] / np.linalg.norm(normal))       # simple head-light shading
        upd = inside & (z < zbuf) & (z > 0)
        zbuf[upd] = z[upd]
        rgb[upd] = np.clip(warped[upd].astype(np.float32) * shade, 0, 255).astype(np.uint8)
    mask = np.isfinite(zbuf)
    depth = np.where(mask, zbuf, 0).astype(np.float32)
    return rgb, depth, mask
