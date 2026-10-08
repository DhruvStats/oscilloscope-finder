# Oscilloscope finder (RTB2004, TDS 2014, TDS 1002)

Upload a photo of a cluttered bench; the PoC oscilloscopes - R&S RTB2004, Tektronix TDS 2014 and
Tektronix TDS 1002 - are found and named, other objects are labelled faintly for context. Everything runs
locally on the CPU — no cloud calls, no downloads at runtime.

- `web/index.html` — the page: upload / drag & drop / paste, focus mode, thresholds, download labelled image.
- `server/app.py` — FastAPI server with the Leonardo AR contract (`GET /health`, `POST /v1/recognitions`).
  It runs two YOLOX-Tiny models: the fine-tuned 3-oscilloscope model (target) and the stock COCO model (context).
- `training/yolox_tiny_osc3.py` — YOLOX experiment (3 classes, 416×416); `yolox_tiny_rtb2004.py` is the earlier 1-class one.
- `training/train_cpu.py` — CPU fine-tuning loop (the official YOLOX trainer needs CUDA).
- `datasets/oscilloscopes3/` — 3-class dataset (`synth/gen3.py`): 1,812 train / 452 validation (80/20) images built
  from ~350 labelled real images (64 lab photos + the 2026-10-08 batch of 39 photos and 3 videos) with 3D renders,
  real cut-outs and crops; `test2017` = 23 real images never used for training (9 from an unseen lab).
  Formats: COCO (`annotations/`), YOLO (`yolo/`), Pascal VOC (`voc/`) - `tools/export_formats.py`.
- `raw/` — the lab photos, `photos_manifest.csv` (instrument + session per photo), `real_labels.json` (checked boxes).
- `tools/` — `draft_real_labels.py` / `finalize_real_labels.py` (labelling), `eval_real.py` (score on real photos),
  `prelabel.py` (Label Studio pre-annotations), `extract_frames.py` (video to frames), `find_duplicates.py`.

## Folder map (C:\Users\WAGH\oscilloscope-detection is the main folder)

| Folder | Content | In git |
|---|---|---|
| `server/`, `web/` | detector API + web page | yes |
| `training/`, `synth/`, `tools/` | training, dataset generator, labelling/eval tools | yes |
| `labelstudio/` | Label Studio setup (project "PoC oscilloscopes") | yes (not its database) |
| `models/deploy/` | deployed weights | yes |
| `raw/`, `datasets/`, `models/training/` | photos, labels, generated datasets, checkpoints | no (lab data) |
| `legacy-yolov8/` | first YOLOv8 + OCR version | code only |
| `media/` | 360/3D videos, interactive viewer and the scripts that made them | no |
| `_archive/` | old copy from karta-ai, kept aside | no |

## Mode: "oscilloscope" (default) or exact model

`LABEL_MODE=generic` (default) answers every target as **Oscilloscope** - on the 23 real test images 30/35
found at 40% (31/35 at 30%), 0 naming errors. The detector's model guess is kept in `model_hint`.
`LABEL_MODE=models` names RTB2004 / TDS 2014 / TDS 1002 (plus the second-stage Tektronix check), ~60% right.

## Current result (real held-out images, scored as the app runs: whole photo + tiles, 40% cut-off)

| model | test set | right | wrong name | missed | false alarms |
|---|---|---|---|---|---|
| v3 | 14 lab photos (18 instruments) | 14 | 4 | 0 | 1 |
| v3 | 23 images incl. 9 from an unseen lab (35 instruments) | 21 | 7 | 7 | 1 |
| **v4 (deployed)** | same 23 images | **21** | 9 | **5** | 3 |

v4 (trained on the 2026-10-08 batch, 1,812 training images) equals v3 at 40% and beats it at 30/50/60%
(22/20/20 vs 21/18/18). Main remaining error: TDS 2014 vs TDS 1002 naming.
Reproduce: `.venv/Scripts/python tools/eval_app.py models/deploy/yolox_tiny_osc3.pth`

## Run

```bash
.venv/Scripts/python -m uvicorn server.app:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001. To use it from a Quest or another PC on the LAN, use `--host 0.0.0.0`
(LAN only, never expose it to the Internet).

The page can also talk to the Leonardo server on the Mac: open **Server** in the page and enter its
address. Responses with `detections[].label/confidence/bbox{x,y,width,height}` (normalised) are understood;
a detection is the target when `target: true` or its label contains "oscillo"/"rtb".

## API response (unchanged Leonardo contract, plus `target`)

```json
{
  "request_id": "…", "session_id": "web-ui", "mode": "real", "simulated": false,
  "device": "cpu", "inference_ms": 420.5, "image": {"width": 1600, "height": 717},
  "detections": [
    {"class_id": "rtb2004:0", "label": "oscilloscope_rtb2004", "confidence": 0.93,
     "bbox": {"x": 0.26, "y": 0.49, "width": 0.59, "height": 0.16}, "target": true},
    {"class_id": "coco:39", "label": "bottle", "confidence": 0.61,
     "bbox": {"x": 0.18, "y": 0.49, "width": 0.10, "height": 0.14}, "target": false}
  ]
}
```

## Retrain

```bash
.venv/Scripts/python synth/gen3.py --per-class 250 --real-labels raw/real_labels.json
.venv/Scripts/python training/train_cpu.py --exp yolox_tiny_osc3 --epochs 20 --init models/training/osc3_renders_only.pth
.venv/Scripts/python tools/eval_real.py models/training/yolox_tiny_osc3/best_ckpt.pth
```

Writes `models/training/yolox_tiny_osc3/{best,last}_ckpt.pth` and `train_log.json`; the server picks it up on restart.
On the NVIDIA workstation the official trainer works with the same experiment file:

```bash
PYTHONPATH=models/yolox python models/yolox/tools/train.py -f training/yolox_tiny_osc3.py -d 1 -b 16 -c models/yolox_tiny.pth
```

## Provenance

- YOLOX source: tag `0.3.0`, commit `419778480ab6ec0590e5d3831b3afb3b46ab2aa3` (Apache-2.0).
- `models/yolox_tiny.pth` SHA-256 `9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0`.
- Training data: lab phone photos, 3D renders built from them, Batronix/Datatec studio images of the RTB2004 and
  one TDS 1002 photo by Berserkerus (Wikimedia Commons, CC BY-SA 2.5) —
  check licences per the Leonardo process before use beyond the PoC.

## Hosted demo (Render)

The repository contains a `Dockerfile` and a `render.yaml` blueprint. In Render: **New > Blueprint**, select this
repository and deploy. The demo is open (no password); to protect it, add a `DEMO_PASSWORD` environment
variable in the Render dashboard. The free plan (512 MB) runs both models: the
3 oscilloscopes and the everyday objects (bottle, chair, laptop ...), about 350-410 MB at peak. Set
`CONTEXT_MODEL=off` (and build with `WITH_CONTEXT=0`) to label only the oscilloscopes. The free plan sleeps when idle, so the first request after a pause
takes about a minute.

Note: the hosted demo processes uploaded photos on Render's cloud servers. This is an approved public demo, separate
from the on-premise Leonardo PoC, which must stay on the local network.

Not in the repository (kept local by `.gitignore`): lab photos, datasets, photo-derived textures, training
checkpoints and the YOLOX source (cloned at build time, pinned to commit `41977848`).

## Labelling with Label Studio

Same approach as the joystick dataset in the Leonardo notes: Label Studio runs locally in its own isolated
environment (`.labelstudio-venv`, like `uv tool install label-studio`), its database lives in
`labelstudio/data/` (git-ignored), and photos are served straight from `raw/` - nothing is copied or uploaded.

| File | Purpose |
|---|---|
| `labelstudio/labeling_config.xml` | interface: 3 box labels (keys 1/2/3) + photo status + notes |
| `labelstudio/GUIDELINES.md` | how to tell the three models apart, what to box and what not |
| `labelstudio/make_tasks.py` | tasks with boxes pre-filled (reviewed boxes, or model proposals for new photos) |
| `labelstudio/setup_project.py` | creates the project, connects `raw/`, imports the tasks |
| `labelstudio/import_export.py` | Label Studio JSON export -> `raw/real_labels.json` |
| `labelstudio/start.ps1` | starts Label Studio on http://localhost:8080 |

```powershell
# one-time install (already done on this PC)
python -m venv .labelstudio-venv; .labelstudio-venv\Scripts\python -m pip install label-studio

.\labelstudio\start.ps1                                   # 1. start, sign up locally at http://localhost:8080
.venv\Scripts\python labelstudio\make_tasks.py --new raw\new_photos   # 2. tasks (+ model boxes for new photos)
$env:LS_TOKEN = "<Account & Settings > token>"            # 3. create the project
.labelstudio-venv\Scripts\python labelstudio\setup_project.py
# 4. label in the browser; then Export > JSON
.venv\Scripts\python labelstudio\import_export.py C:\path\to\export.json   # 5. back into the dataset
.venv\Scripts\python synth\gen3.py --per-class 300 --real-labels raw\real_labels.json   # 6. rebuild + retrain
```

After step 5 do not run `tools/finalize_real_labels.py` again - it would rebuild `real_labels.json` from the
older draft and overwrite the Label Studio corrections.

## Confidence threshold

The detector gives every candidate box a confidence score; most candidates are noise (sockets, signs,
shadows) with low scores. One cut-off is therefore required, but it is fixed on the server
(`TARGET_MIN_CONF`, default 0.4, chosen on the real held-out photos with `tools/eval_app.py`) instead of a
slider in the page.

## Long-term workflow (self-hosted, data stays in the lab)

| Need | How | Files |
|---|---|---|
| Add instruments | one registry: names, colours, sizes, weights, 3D textures; everything reads it | `config/instruments.yaml`, `tools/registry_sync.py` |
| Team labelling | Label Studio on the lab LAN, invite-only sign-up | `labelstudio/start_team.ps1` |
| Fast retraining | GPU used automatically; one-zip bundle for the NVIDIA workstation | `training/train_cpu.py --device auto`, `tools/package_for_workstation.py` |
| Learning from real use | opt-in capture of unsure frames + "Report wrong result" -> Label Studio | `CAPTURE_MODE=on`, `labelstudio/add_tasks.py` |
| Edge / mobile / headset | ONNX export, verified against PyTorch | `tools/export_onnx.py` -> `models/export/` |

Adding an instrument: add it to `config/instruments.yaml` -> `python tools/registry_sync.py` (Label Studio
interface) -> photograph it into `raw/new_photos/<label>/` -> label in Label Studio -> import -> rebuild the
dataset -> retrain (workstation) -> `tools/eval_app.py` -> deploy if better.

Learning from real use (lab server only, off by default, never on Render):
```powershell
$env:CAPTURE_MODE = "on"; .venv\Scripts\python -m uvicorn server.app:app --host 0.0.0.0 --port 8011
# clients opt in (page checkbox / Unity "Contribute Unsure Frames"); frames land in raw/captures/<date>/
.venv\Scripts\python labelstudio\make_tasks.py --new raw\captures\2026-10-08 --only-new
$env:LS_TOKEN = "<token>"; .labelstudio-venv\Scripts\python labelstudio\add_tasks.py
```

## Optional: Roboflow connector (off by default)

`tools/roboflow_connector.py` (run with `.roboflow-venv`, Roboflow's official SDK in its own environment):
`status`, `push` (dataset to a Roboflow project, for team labelling or training Roboflow models), `pull`
(a dataset version edited in Roboflow back into `raw/real_labels.json`), `eval` (score a Roboflow-trained model on
our 23 real test images with the same metric as `tools/eval_app.py`). Anything that sends images is a dry run
unless `ROBOFLOW_API_KEY` is set, `ROBOFLOW_UPLOAD_APPROVED=yes` (only after Leonardo approves cloud upload)
and `--send` is given. Free Roboflow projects are public - use a paid/private workspace for lab photos.
