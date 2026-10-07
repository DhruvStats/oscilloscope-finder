# Oscilloscope finder (RTB2004, TDS 2014, TDS 1002)

Upload a photo of a cluttered bench; the PoC oscilloscopes - R&S RTB2004, Tektronix TDS 2014 and
Tektronix TDS 1002 - are found and named, other objects are labelled faintly for context. Everything runs
locally on the CPU — no cloud calls, no downloads at runtime.

- `web/index.html` — the page: upload / drag & drop / paste, focus mode, thresholds, download labelled image.
- `server/app.py` — FastAPI server with the Leonardo AR contract (`GET /health`, `POST /v1/recognitions`).
  It runs two YOLOX-Tiny models: the fine-tuned 3-oscilloscope model (target) and the stock COCO model (context).
- `training/yolox_tiny_osc3.py` — YOLOX experiment (3 classes, 416×416); `yolox_tiny_rtb2004.py` is the earlier 1-class one.
- `training/train_cpu.py` — CPU fine-tuning loop (the official YOLOX trainer needs CUDA).
- `datasets/oscilloscopes3/` — 3-class COCO dataset (`synth/gen3.py`): 3D renders + real cut-outs pasted on
  bench scenes + 40 real train photos; `test2017` = 12 real photos never used for training.
- `raw/` — the lab photos, `photos_manifest.csv` (instrument + session per photo), `real_labels.json` (checked boxes).
- `tools/` — `draft_real_labels.py` / `finalize_real_labels.py` (labelling), `eval_real.py` (score on real photos),
  `prelabel.py` (Label Studio pre-annotations), `extract_frames.py` (video to frames), `find_duplicates.py`.

## Current result (real held-out photos)

| | renders only | + real cut-outs |
|---|---|---|
| correctly found and named | 8/15 | 10/15 |
| false alarms | 9 | 0 |
| AP50 | 0.60 | 0.74 |

Weak spot: TDS 2014 fronts are sometimes named TDS 1002 (too few real TDS 2014 photos). More real photos of the
TDS 2014 front, from other places and distances, is the most valuable next data.

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
- Training data: lab phone photos, 3D renders built from them, and Batronix studio images of the RTB2004 —
  check licences per the Leonardo process before use beyond the PoC.

## Hosted demo (Render)

The repository contains a `Dockerfile` and a `render.yaml` blueprint. In Render: **New > Blueprint**, select this
repository, set `DEMO_PASSWORD` when asked (recommended), and deploy. The free plan (512 MB) runs the 3-oscilloscope
model only (`CONTEXT_MODEL=off`, about 350 MB in use); on Standard or larger, set `CONTEXT_MODEL=on` and build with
`WITH_CONTEXT=1` to also label everyday objects. The free plan sleeps when idle, so the first request after a pause
takes about a minute.

Note: the hosted demo processes uploaded photos on Render's cloud servers. This is an approved public demo, separate
from the on-premise Leonardo PoC, which must stay on the local network.

Not in the repository (kept local by `.gitignore`): lab photos, datasets, photo-derived textures, training
checkpoints and the YOLOX source (cloned at build time, pinned to commit `41977848`).
