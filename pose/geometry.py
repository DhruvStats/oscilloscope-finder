"""Frames, projection and the OpenCV -> Unity conversion used by the pose stage.

Conventions (one place, used everywhere):

  Object (mesh) frame - right-handed, metres, origin at the centre of the instrument's box:
      +x  to the instrument's right (seen from the front)   +y  up   +z  out of the front panel
  Camera frame (OpenCV / FoundationPose) - right-handed:  +x right  +y down  +z forward (viewing direction)
  A pose ``T`` (4x4) maps object points to camera points: p_cam = T[:3,:3] @ p_obj + T[:3,3]   ("ob_in_cam")

  Unity - left-handed: camera +x right, +y up, +z forward. Unity's OBJ importer mirrors x, so a mesh vertex
  (x, y, z) appears in the imported prefab at (-x, y, z). For a prefab built from our OBJ:
      position_unity = S @ t                 rotation_unity = S @ R @ F
  with S = diag(1, -1, 1) (camera: y down -> y up) and F = diag(-1, 1, 1) (importer mirror); det = +1, so it
  is a proper rotation. tests/test_pose_geometry.py checks this on box corners.
"""
import numpy as np

S = np.diag([1.0, -1.0, 1.0])
F = np.diag([-1.0, 1.0, 1.0])


def box_corners(size_m):
    """8 corners of a box with size (width, height, depth) in metres, centred at the origin (object frame)."""
    w, h, d = (s / 2 for s in size_m)
    return np.array([[x, y, z] for x in (-w, w) for y in (-h, h) for z in (-d, d)], float)


def transform(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def project(K, pts_cam):
    """Pixel coordinates of camera-frame points (N x 3) -> N x 2."""
    uvw = pts_cam @ K.T
    return uvw[:, :2] / uvw[:, 2:3]


def projected_box(K, T, size_m):
    """Axis-aligned 2D box [x0, y0, x1, y1] (pixels) of the posed 3D box."""
    uv = project(K, transform(T, box_corners(size_m)))
    return np.concatenate([uv.min(0), uv.max(0)])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)


def default_intrinsics(width, height, hfov_deg=70.0):
    """Pinhole guess when the client sends no calibration (good enough for the CPU fallback, not for accuracy)."""
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return intrinsics(f, f, width / 2, height / 2)


def quat_from_matrix(R):
    """Rotation matrix -> quaternion (x, y, z, w), the order Unity's Quaternion constructor uses."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def to_unity(T):
    """ob_in_cam (OpenCV) -> pose of the imported prefab in Unity camera space: position (m) and quaternion."""
    R_u = S @ T[:3, :3] @ F
    t_u = S @ T[:3, 3]
    return {"position": [round(float(v), 5) for v in t_u],
            "rotation": [round(float(v), 6) for v in quat_from_matrix(R_u)]}


def rotation_matrix(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def facing_camera(t, yaw_deg=0.0, pitch_deg=0.0):
    """Pose of an upright instrument whose front faces the camera, at translation t (camera frame).

    Object +y (up) -> camera -y (camera y points down); object +z (out of the front) -> camera -z (towards the
    camera). yaw turns the instrument about its vertical axis, pitch tilts the camera down onto it.
    """
    base = np.diag([1.0, -1.0, -1.0])
    # R_x(+a) turns the instrument's up vector (camera -y) towards the camera (-z): the top becomes visible
    R = rotation_matrix("x", pitch_deg) @ base @ rotation_matrix("y", yaw_deg)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
