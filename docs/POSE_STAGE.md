# Pose stage: Recognition -> FoundationPose -> Unity

The middle block of the pipeline: after recognition says *which* oscilloscope is in the frame (label, box,
confidence), the pose stage selects that model's 3D mesh and estimates its **6D pose** (position + orientation),
first by registration, then by tracking. The pose goes to Unity, where it is applied to the reference object
(prefab with the local AR points) of the same model.

```mermaid
flowchart LR
  C[Quest / client<br/>RGB frame (+ depth) + intrinsics<br/>+ camera pose at capture] -->|POST /v1/pose| R[Recognition<br/>YOLOX + Tektronix check]
  R -->|label + box| S[PoseService<br/>pose/service.py]
  REG[(config/instruments.yaml<br/>label -> mesh, size)] --> S
  S -->|1st frame| M[Mask from box<br/>SAM 2 / GrabCut / box]
  M --> FP[FoundationPose<br/>register]
  S -->|next frames| FT[FoundationPose<br/>track_one]
  FP --> O[ob_in_cam 4x4 + Unity pose]
  FT --> O
  O --> U[Unity ServerPoseProvider<br/>-> reference object of that model]
```

## How FoundationPose fits into the current server

| FoundationPose needs | Where it comes from in our system |
|---|---|
| which object / which mesh | recognition label -> `mesh` in `config/instruments.yaml` (`models/meshes/<label>/<label>.obj`) |
| metric, textured mesh | `tools/build_meshes.py`: box of the real size (datasheet) with the six face photos, one texture atlas |
| object mask (first frame) | detector box -> `pose/masks.py` (SAM 2 with box prompt on the GPU server, GrabCut on CPU, or the box) |
| RGB image | the frame the client already sends |
| depth in metres, aligned to RGB | new: optional `depth` upload in `/v1/pose` (16-bit PNG in mm, or `.npy` in m) |
| camera intrinsics K | new: `fx, fy, cx, cy` fields in `/v1/pose` (Quest: Passthrough Camera API helpers) |
| GPU, CUDA, its own build | runs on the LAN GPU workstation in the official Docker image (`pose/gpu/setup_foundationpose.sh`) |

The official code is used unchanged: `pose/backends.py` imports its `estimater` module from `FOUNDATIONPOSE_DIR`
and calls exactly what `run_demo.py` calls - `FoundationPose(model_pts, model_normals, mesh, scorer, refiner,
glctx)`, `est.register(K, rgb, depth, ob_mask, iteration=5)` and `est.track_one(rgb, depth, K, iteration=2)`.
One estimator is kept per instrument; one tracking state per client session.

**Tracking logic** (`pose/service.py`): register on the first frame (or after a reset / model change); then track.
After every frame the posed 3D box is projected into the image and compared with the detector's box; if they no
longer overlap (IoU < 0.25) the track is "lost" and the next frame registers again. Without a detection,
FoundationPose keeps tracking; the CPU fallback cannot.

**Backends**: `POSE_BACKEND=auto` uses FoundationPose when `FOUNDATIONPOSE_DIR` is set and CUDA is available,
otherwise the **geometric CPU fallback** (position from depth, or without depth from the known instrument width and
the box width; orientation assumed "upright, front facing the camera"). The fallback is approximate - it exists so
the Unity side and the whole chain can be tested today without a GPU; `"approximate": true` marks its answers.

## Conventions (important for Unity)

- **Object (mesh) frame**: metres, origin at the centre of the instrument's box, **+x right, +y up, +z out of the
  front panel** (right-handed). Sizes: RTB2004 0.390 x 0.220 x 0.152 m, TDS 2014 / TDS 1002 0.324 x 0.151 x 0.124 m.
- **`ob_in_cam`**: 4x4, row-major, maps object points into the OpenCV camera frame (x right, y down, z forward).
- **`unity`**: the same pose for a prefab created by importing our `.obj` into Unity (Unity mirrors x on import),
  in Unity camera space (x right, y up, z forward): `position` (m) and `rotation` quaternion `[x, y, z, w]`.
  World pose = camera pose **at capture time** x this local pose (`ServerPoseProvider.cs` does it).
  The conversion is checked by `tests/test_pose.py` (every box corner lands in the same place).

## API: `POST /v1/pose` (multipart form)

| field | | |
|---|---|---|
| `image` | required | JPEG/PNG frame |
| `depth` | optional | aligned depth: 16-bit PNG (units x `depth_scale`, default mm) or `.npy` float metres; **required by FoundationPose** |
| `fx, fy, cx, cy` | recommended | intrinsics of the sent image (pixels); guessed (70 deg FOV) when missing |
| `session_id` | recommended | one tracking state per client |
| `label` | optional | pose only this model (e.g. `tek_tds1002`); default: best recognised instrument with a mesh |
| `reset` | optional | `true` = forget the track and register again |

Response: the recognition `detections` plus `pose`:
```json
"pose": {"state": "registered | tracking | lost | no_target | needs_depth", "label": "tek_tds1002",
         "backend": "foundationpose", "approximate": false,
         "ob_in_cam": [16 numbers], "frame": "opencv_camera",
         "unity": {"position": [x, y, z], "rotation": [x, y, z, w]},
         "score": 0.93, "box_agreement": 0.86, "mask": "sam2", "mesh": "models/meshes/tek_tds1002/tek_tds1002.obj",
         "size_m": [0.324, 0.151, 0.124], "pose_ms": 41.0}
```
`GET /health` reports `pose_backend` (and `pose_error` if the requested backend could not start).

## Run it

CPU (this PC, any laptop) - the whole chain with the approximate backend:
```bash
.venv/Scripts/python tools/build_meshes.py                       # meshes + models/meshes/preview.jpg
.venv/Scripts/python tools/make_pose_testset.py tek_tds1002      # synthetic RGB-D with ground truth
.venv/Scripts/python tools/pose_eval.py datasets/pose_test/tek_tds1002 --overlay
.venv/Scripts/python tests/test_pose.py
```
GPU workstation - FoundationPose: `pose/gpu/setup_foundationpose.sh` (official Docker image, weights, the official
demo on our scene, then `tools/pose_eval.py --backend foundationpose`, then the server with
`POSE_BACKEND=foundationpose`). `tools/package_for_workstation.py` packs the meshes, textures and test scenes.

Unity: add `unity/ServerPoseProvider.cs`, set the server URL and the intrinsics, and bind each label to its
reference object. Its `onPose(label, worldPosition, worldRotation)` event has the same shape as a simulated pose
provider, so the AR content does not change when the simulated provider is swapped for the server.

## First test plan (one model: TDS 1002)

1. Workstation: official demo + `pose_eval.py --backend foundationpose` on `datasets/pose_test/tek_tds1002`
   (synthetic, ground truth known) - expect a few mm / degrees.
2. Real RGB-D: record the TDS 1002 with an RGB-D camera (e.g. RealSense) in the official layout (`rgb/`, `depth/`,
   `cam_K.txt`), first mask from our detector box; run `pose_eval.py` / the demo and check the overlay.
3. Live: server with `POSE_BACKEND=foundationpose`, client sends RGB + depth + K; Unity places the TDS 1002 prefab.
4. Then RTB2004 and TDS 2014 (same pipeline, other mesh).

## Status (CPU, synthetic TDS 1002 sequence, 40 frames)

| backend | position error | orientation error | notes |
|---|---|---|---|
| geometric, with depth | ~1 cm | ~28 deg (assumes front facing) | for integration tests only |
| geometric, no depth | ~2 cm | ~28 deg | what the Quest can do today without depth |
| FoundationPose | to be measured on the GPU workstation | | the target backend |

Open points: depth on the Quest 3 (no camera-aligned depth today: external RGB-D camera for the PoC, then stereo
from the two passthrough cameras); box meshes are approximations (CAD or a scan would improve FoundationPose);
FoundationPose licence is research/evaluation only.
