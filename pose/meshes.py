"""Metric, textured box meshes of the instruments for FoundationPose (and for the Unity reference objects).

    .venv/Scripts/python tools/build_meshes.py           -> models/meshes/<label>/<label>.obj/.mtl/.png

Each instrument is close to a box, so a first mesh is a box of the real size (config/instruments.yaml:
width_mm / height_mm / depth_mm) with the six face photos the dataset generator already uses
(synth/textures, first texture variant). The six faces are packed into ONE texture atlas, because
FoundationPose loads the mesh with trimesh and needs a single textured mesh, not a multi-material scene.

Object frame (pose/geometry.py): metres, origin at the box centre, +x right, +y up, +z out of the front.
A manufacturer CAD model or a photogrammetry scan can replace the box later: same frame, same file name.
"""
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "synth", "textures")
CELL = 512                       # atlas: 3 x 2 cells of CELL x CELL px

# corners of each face in texture order top-left, top-right, bottom-right, bottom-left (as seen from outside,
# texture upright) - the same assignment as synth/gen3.py box_corners, with y pointing up instead of down.
# Corner = signs of (x, y, z): x right, y up, z front.
FACES = {
    "front":  [(-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1)],
    "back":   [(1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1)],
    "right":  [(1, 1, 1), (1, 1, -1), (1, -1, -1), (1, -1, 1)],
    "left":   [(-1, 1, -1), (-1, 1, 1), (-1, -1, 1), (-1, -1, -1)],
    "top":    [(-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)],
    "bottom": [(-1, -1, 1), (1, -1, 1), (1, -1, -1), (-1, -1, -1)],
}
ORDER = ["front", "back", "right", "left", "top", "bottom"]


def size_m(inst):
    """(width, height, depth) in metres from a registry entry."""
    if not inst.get("height_mm") or not inst.get("depth_mm"):
        raise ValueError(f"{inst['label']}: height_mm and depth_mm are needed in config/instruments.yaml")
    return (inst["width_mm"] / 1000, inst["height_mm"] / 1000, inst["depth_mm"] / 1000)


def mesh_path(inst):
    return os.path.join(ROOT, inst.get("mesh") or f"models/meshes/{inst['label']}/{inst['label']}.obj")


def build(inst, out_dir=None):
    """Write <label>.obj, <label>.mtl and <label>.png (atlas). Returns the .obj path."""
    label = inst["label"]
    obj = mesh_path(inst) if out_dir is None else os.path.join(out_dir, f"{label}.obj")
    os.makedirs(os.path.dirname(obj), exist_ok=True)
    stem = os.path.splitext(obj)[0]
    variant = inst["textures"][0] if inst.get("textures") else None
    atlas = np.full((2 * CELL, 3 * CELL, 3), 128, np.uint8)
    for k, face in enumerate(ORDER):
        if variant:
            img = cv2.imread(os.path.join(TEX, variant[face] + ".png"), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"{label}: texture {variant[face]}.png not found in synth/textures")
            r, c = divmod(k, 3)
            atlas[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = cv2.resize(img, (CELL, CELL),
                                                                                 interpolation=cv2.INTER_AREA)
    cv2.imwrite(stem + ".png", atlas)

    hw, hh, hd = (s / 2 for s in size_m(inst))
    v, vt, f = [], [], []
    eps = 0.5 / CELL                 # keep UVs inside the cell (no bleeding from the neighbour)
    for k, face in enumerate(ORDER):
        r, c = divmod(k, 3)
        u0, u1 = c / 3 + eps / 3, (c + 1) / 3 - eps / 3
        # OBJ texture v runs upwards from the bottom of the image; atlas row 0 is the top half
        v1, v0 = 1 - r / 2 - eps / 2, 1 - (r + 1) / 2 + eps / 2
        base = len(v)
        for sx, sy, sz in FACES[face]:
            v.append((sx * hw, sy * hh, sz * hd))
        vt += [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]
        # counter-clockwise seen from outside: TL, BL, BR and TL, BR, TR
        f.append((base + 1, base + 4, base + 3))
        f.append((base + 1, base + 3, base + 2))
    name = os.path.basename(stem)
    with open(stem + ".mtl", "w") as m:
        m.write(f"newmtl {name}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd {name}.png\n")
    with open(obj, "w") as o:
        o.write(f"# {inst.get('display_name', label)} - box {2 * hw:.4f} x {2 * hh:.4f} x {2 * hd:.4f} m "
                f"(x right, y up, z out of the front; origin at the centre)\nmtllib {name}.mtl\nusemtl {name}\n")
        o.writelines(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in v)
        o.writelines(f"vt {a:.5f} {b:.5f}\n" for a, b in vt)
        o.writelines(f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in f)
    return obj


def load_obj(path):
    """Minimal reader for our own box OBJ (vertices, uvs, faces, texture) - no trimesh needed on the CPU side."""
    v, vt, faces = [], [], []
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            v.append([float(x) for x in p[1:4]])
        elif p[0] == "vt":
            vt.append([float(x) for x in p[1:3]])
        elif p[0] == "f":
            faces.append([[int(i) - 1 for i in q.split("/")[:2]] for q in p[1:4]])
    tex = cv2.imread(os.path.splitext(path)[0] + ".png", cv2.IMREAD_COLOR)
    return np.array(v), np.array(vt), np.array(faces), tex
