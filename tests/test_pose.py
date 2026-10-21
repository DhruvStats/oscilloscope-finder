"""Checks for the pose stage (run: .venv/Scripts/python -m pytest tests/test_pose.py -q, or directly)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pose import geometry as G  # noqa: E402


def random_pose(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, rng.uniform([-0.5, -0.5, 0.5], [0.5, 0.5, 2.0])
    return T


def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_unity_conversion_moves_every_corner_to_the_same_place():
    """A corner of the imported prefab (Unity mirrors x on OBJ import) placed with the Unity pose must land where
    the OpenCV pose puts the same physical corner, seen in Unity's y-up camera space."""
    rng = np.random.default_rng(1)
    corners = G.box_corners((0.324, 0.151, 0.124))
    for _ in range(50):
        T = random_pose(rng)
        u = G.to_unity(T)
        R_u, t_u = quat_to_matrix(u["rotation"]), np.array(u["position"])
        assert abs(np.linalg.det(R_u) - 1) < 1e-4       # quaternion rounded to 6 decimals
        for p in corners:
            in_unity = R_u @ (G.F @ p) + t_u          # prefab vertex = mirrored mesh vertex
            expected = G.S @ (T[:3, :3] @ p + T[:3, 3])
            assert np.allclose(in_unity, expected, atol=1e-4)


def test_quaternion_roundtrip():
    rng = np.random.default_rng(2)
    for _ in range(50):
        R = random_pose(rng)[:3, :3]
        assert np.allclose(quat_to_matrix(G.quat_from_matrix(R)), R, atol=1e-6)


def test_facing_camera_shows_front_and_top():
    T = G.facing_camera(np.array([0, 0, 1.0]), 0, 20)
    R = T[:3, :3]
    assert (R @ [0, 0, 1])[2] < 0          # the front (+z) points at the camera
    assert (R @ [0, 1, 0])[1] < 0          # up stays up in the image (camera y points down)
    assert (R @ [0, 1, 0])[2] < 0          # with pitch the top tilts towards the camera


def test_geometric_pose_on_synthetic_render():
    """End to end on CPU: render a mesh at a known pose, estimate it back from RGB-D."""
    import cv2
    from config import registry
    from pose import meshes
    from pose.render import render
    from pose.service import PoseService
    insts = registry.load()
    inst = {i["label"]: i for i in insts}["tek_tds1002"]
    obj = meshes.mesh_path(inst)
    if not os.path.exists(obj):
        meshes.build(inst)
    size, K = meshes.size_m(inst), G.intrinsics(600, 600, 320, 240)
    gt = G.facing_camera(np.array([0.05, 0.02, 0.9]), 0, 10)
    rgb, depth, mask = render(os.path.splitext(obj)[0] + ".png", size, gt, K, (480, 640),
                              background=np.full((480, 640, 3), 90, np.uint8))
    ys, xs = np.nonzero(mask)
    det = {"label": "tek_tds1002", "confidence": 0.9, "target": True, "model_certain": True,
           "bbox": {"x": xs.min() / 640, "y": ys.min() / 480, "width": np.ptp(xs) / 640, "height": np.ptp(ys) / 480}}
    r = PoseService(insts, backend="geometric", mask_method="box").process(rgb, depth, K, [det], "t")
    T = np.array(r["ob_in_cam"]).reshape(4, 4)
    assert r["state"] == "registered" and r["box_agreement"] > 0.6
    assert np.linalg.norm(T[:3, 3] - gt[:3, 3]) < 0.03         # within 3 cm
    assert cv2 is not None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
