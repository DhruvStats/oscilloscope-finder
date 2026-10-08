"""Oscilloscope finder: local YOLOX-Tiny inference behind the Leonardo AR recognition contract.

Two on-premise models run on the CPU:
  - target    fine-tuned YOLOX-Tiny for the PoC oscilloscopes: R&S RTB2004, Tektronix TDS 2014, TDS 1002
              (falls back to the RTB2004-only model if the 3-class weights are not trained yet)
  - coco      stock YOLOX-Tiny COCO checkpoint, for context objects (bottle, chair, laptop ...)

    .venv/Scripts/python -m uvicorn server.app:app --port 8001

Environment (all optional):
  TARGET_EXP / TARGET_CKPT   which experiment / weights to load (default: first trained one found)
  CONTEXT_MODEL=off          skip the COCO context model (saves ~150 MB RAM, e.g. on small cloud plans)
  COCO_CKPT                  path of the COCO checkpoint (default models/yolox_tiny.pth)
  TORCH_THREADS              CPU threads for inference (default: half the cores)
  MAX_UPLOAD_MB              reject larger uploads (default 15)
  TILED_DETECTION=off        only look at the whole photo (faster, but misses small, distant instruments)
  LABEL_MODE=models          name the exact model instead of just "oscilloscope" (default: generic)
  TEK_CHECK=off              skip the second-stage TDS 2014 / TDS 1002 classifier (models/deploy/tek_classifier.pth)
  TEK_MIN_CONF               classifier confidence needed to (re)name a Tektronix (default 0.6)
  CONTEXT_TILES=off          everyday objects from the whole photo only (faster)
  DEMO_PASSWORD              if set, the page and API ask for HTTP basic auth (any user name, this password)
  CAPTURE_MODE=on            lab server only: store unsure frames from opted-in clients and reported photos
                             in raw/captures/<date>/ for labelling (off by default, always off on Render)
"""
import base64
import datetime
import json
import secrets
import os
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models", "yolox"))
sys.path.insert(0, os.path.join(ROOT, "training"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from yolox.data.data_augment import ValTransform  # noqa: E402
from yolox.data.datasets import COCO_CLASSES  # noqa: E402
from yolox.exp import get_exp  # noqa: E402
from yolox.utils import postprocess  # noqa: E402

import importlib  # noqa: E402

sys.path.insert(0, ROOT)
from config import registry  # noqa: E402

# instrument names, display names and colours: config/instruments.yaml (order = model output order)
INSTRUMENTS = registry.load()
DISPLAY_NAMES = {it["label"]: it["display_name"] for it in INSTRUMENTS}
COLOURS = {it["label"]: it["colour"] for it in INSTRUMENTS}
# first available wins; override with TARGET_EXP / TARGET_CKPT
TARGET_MODELS = [
    ("yolox_tiny_osc3", [it["label"] for it in INSTRUMENTS]),
    ("yolox_tiny_rtb2004", ["rs_rtb2004"]),
]
COCO_CKPT = os.environ.get("COCO_CKPT", os.path.join(ROOT, "models", "yolox_tiny.pth"))
# Confidence cut-offs. The model scores every candidate box; below these the boxes are mostly noise
# (sockets, signs, shadows). Fixed values chosen on the real held-out photos (tools/eval_app.py) so users
# do not have to tune anything; override with the environment variables if needed.
TARGET_MIN_CONF = float(os.environ.get("TARGET_MIN_CONF", "0.4"))
COCO_MIN_CONF = float(os.environ.get("COCO_MIN_CONF", "0.35"))
TILES = os.environ.get("TILED_DETECTION", "on").lower() not in ("off", "0", "false", "no")
USE_CONTEXT = os.environ.get("CONTEXT_MODEL", "on").lower() not in ("off", "0", "false", "no")
# zoomed tiles for everyday objects too: finds small ones (cups, phones, bottles) in wide shots
CONTEXT_TILES = os.environ.get("CONTEXT_TILES", "on").lower() not in ("off", "0", "false", "no")
MAX_UPLOAD = int(float(os.environ.get("MAX_UPLOAD_MB", "15")) * 1024 * 1024)
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "")
# Learning from real use (opt-in). Frames are NOT stored unless CAPTURE_MODE=on on a lab server; never on the
# hosted demo. Stored: frames a consenting client sends where the model is unsure, and photos reported wrong.
CAPTURE = os.environ.get("CAPTURE_MODE", "off").lower() == "on" and not os.environ.get("RENDER")
CAPTURE_ROOT = os.path.join(ROOT, "raw", "captures")
UNSURE_RANGE = (0.15, 0.6)   # a target box in this confidence band = "model is unsure" -> worth labelling
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", max(1, (os.cpu_count() or 2) // 2))))


class Detector:
    def __init__(self, exp, ckpt, names, prefix, class_agnostic=False):
        self.exp, self.names, self.prefix, self.class_agnostic = exp, names, prefix, class_agnostic
        self.ckpt = ckpt
        self.model = exp.get_model().eval()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        self.tf = ValTransform(legacy=False)

    def detect(self, img, min_conf, tiles=True):
        """Whole image plus overlapping tiles, merged.

        The model sees a 416x416 version of the photo, so a scope that fills 1/8 of a wide room shot
        shrinks to ~50 px and is missed. Tiles along the long side (and a 2x2 grid for big photos) give
        small instruments enough pixels; duplicates across tiles are merged afterwards.
        """
        dets = self(img, min_conf)
        h, w = img.shape[:2]
        if not tiles or max(h, w) < 900:
            return dets
        ov = 0.25
        if w >= h:
            tw, th = int(w * (0.5 + ov / 2)), h
        else:
            tw, th = w, int(h * (0.5 + ov / 2))
        grid = [(0, 0), (w - tw, h - th)]
        if min(h, w) >= 1000:   # big photo: 2x2 grid of 62% tiles
            tw, th = int(w * 0.62), int(h * 0.62)
            grid = [(0, 0), (w - tw, 0), (0, h - th), (w - tw, h - th)]
        for x0, y0 in grid:
            for d in self(img[y0:y0 + th, x0:x0 + tw], min_conf):
                b = d["bbox"]
                bx0, by0 = x0 + b["x"] * tw, y0 + b["y"] * th
                bx1, by1 = bx0 + b["width"] * tw, by0 + b["height"] * th
                # a box touching a seam inside the photo is a cut-off fragment: the other tile or the
                # whole-image pass sees that object complete
                m = 0.02 * max(tw, th)
                if (bx0 - x0 < m and x0 > 0) or (y0 > 0 and by0 - y0 < m) or \
                   (x0 + tw < w and x0 + tw - bx1 < m) or (y0 + th < h and y0 + th - by1 < m):
                    continue
                dets.append(dict(d, bbox={"x": round(bx0 / w, 5), "y": round(by0 / h, 5),
                                          "width": round((bx1 - bx0) / w, 5), "height": round((by1 - by0) / h, 5)}))
        if self.class_agnostic:
            return merge(dets)
        # everyday objects: a cup on a table is fine, so only merge duplicates of the same class
        by_label = {}
        for d in dets:
            by_label.setdefault(d["label"], []).append(d)
        return [k for group in by_label.values() for k in merge(group)]

    @torch.no_grad()
    def __call__(self, img, min_conf):
        h, w = img.shape[:2]
        size = self.exp.test_size
        ratio = min(size[0] / h, size[1] / w)
        x, _ = self.tf(img, None, size)
        out = postprocess(self.model(torch.from_numpy(x).unsqueeze(0).float()),
                          self.exp.num_classes, min_conf, self.exp.nmsthre, self.class_agnostic)[0]
        dets = []
        if out is None:
            return dets
        for x0, y0, x1, y1, obj, cls_conf, cls in out.numpy():
            x0, y0, x1, y1 = np.clip(np.array([x0, y0, x1, y1]) / ratio, 0, [w, h, w, h])
            cls = int(cls)
            dets.append({
                "class_id": f"{self.prefix}:{cls}",
                "label": self.names[cls],
                "display_name": DISPLAY_NAMES.get(self.names[cls], self.names[cls]),
                "confidence": round(float(obj * cls_conf), 4),
                "bbox": {"x": round(x0 / w, 5), "y": round(y0 / h, 5),
                         "width": round((x1 - x0) / w, 5), "height": round((y1 - y0) / h, 5)},
            })
        return dets


def _covered(a, b):
    """Fraction of box a (normalised x, y, width, height) that lies inside box b."""
    ix = max(0.0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
    return ix * iy / max(1e-9, a["width"] * a["height"])


def merge(dets, iou_thr=0.3, contain_thr=0.5):
    """One box per object: highest confidence first, drop boxes that overlap or sit inside a kept one."""
    kept = []
    for d in sorted(dets, key=lambda d: -d["confidence"]):
        b = d["bbox"]
        x0, y0, x1, y1 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]
        area = max(1e-9, b["width"] * b["height"])
        dup = False
        for k in kept:
            kb = k["bbox"]
            ix = max(0, min(x1, kb["x"] + kb["width"]) - max(x0, kb["x"]))
            iy = max(0, min(y1, kb["y"] + kb["height"]) - max(y0, kb["y"]))
            inter = ix * iy
            union = area + kb["width"] * kb["height"] - inter
            if inter / union > iou_thr or inter / area > contain_thr:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def load():
    target = None
    wanted = os.environ.get("TARGET_EXP")
    for exp_name, names in TARGET_MODELS:
        if wanted and exp_name != wanted:
            continue
        exp = importlib.import_module(exp_name).Exp()
        # the deployed, tested weights first; a training checkpoint only as a fallback (it may be a half-finished run)
        deployed = os.path.join(ROOT, "models", "deploy", f"{exp_name}.pth")
        ckpt = os.environ.get("TARGET_CKPT") or (
            deployed if os.path.exists(deployed) else os.path.join(exp.output_dir, exp.exp_name, "best_ckpt.pth"))
        if os.path.exists(ckpt):
            # one physical object gets one identity: suppress overlapping boxes across classes
            target = Detector(exp, ckpt, names, exp_name.replace("yolox_tiny_", ""), class_agnostic=True)
            break
    context = None
    if USE_CONTEXT and os.path.exists(COCO_CKPT):
        context = Detector(get_exp(None, "yolox-tiny"), COCO_CKPT, list(COCO_CLASSES), "coco")
    return target, context


class TekClassifier:
    """Second stage: re-checks every Tektronix box on a full-resolution crop (TDS 2014 vs TDS 1002).

    The two models share one case; only the front (buttons, inputs) or the back module tells them apart, which
    the detector sees at 416 px only. When the classifier is not sure (e.g. a side or top view), the answer
    keeps the detector's name but says so ("model_certain": false) instead of being confidently wrong.
    """

    def __init__(self, path):
        import torchvision
        ck = torch.load(path, map_location="cpu")
        self.classes, self.size = ck["classes"], ck["size"]
        self.mean = np.array(ck["mean"], np.float32)
        self.std = np.array(ck["std"], np.float32)
        m = torchvision.models.mobilenet_v3_small(weights=None)
        m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, len(self.classes))
        m.load_state_dict(ck["model"])
        self.model = m.eval()

    @torch.no_grad()
    def __call__(self, img, bbox):
        H, W = img.shape[:2]
        x, y, w, h = bbox["x"] * W, bbox["y"] * H, bbox["width"] * W, bbox["height"] * H
        x0, y0 = max(0, int(x - 0.1 * w)), max(0, int(y - 0.1 * h))
        x1, y1 = min(W, int(x + 1.1 * w)), min(H, int(y + 1.1 * h))
        crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        if crop.size == 0:
            return None, 0.0
        s = self.size / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                          interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        pad = np.full((self.size, self.size, 3), 114, np.uint8)
        oy, ox = (self.size - crop.shape[0]) // 2, (self.size - crop.shape[1]) // 2
        pad[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        x = (pad.astype(np.float32) / 255 - self.mean) / self.std
        p = torch.softmax(self.model(torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)), 1)[0]
        k = int(p.argmax())
        return self.classes[k], float(p[k])


def apply_tek_check(img, dets):
    """Second-stage naming for Tektronix boxes (in place)."""
    if TEK is None:
        return
    for d in dets:
        if not d["label"].startswith("tek_"):
            continue
        cls, p = TEK(img, d["bbox"])
        if cls is None:
            continue
        d["first_stage_label"] = d["label"]
        d["model_check"] = {"label": cls, "confidence": round(p, 3)}
        if p >= TEK_MIN_CONF:
            d["label"], d["display_name"] = cls, DISPLAY_NAMES.get(cls, cls)
            d["class_id"] = d["class_id"].split(":")[0] + ":" + str(TARGET.names.index(cls))
            d["model_certain"] = True
        else:
            d["model_certain"] = False
            d["display_name"] = DISPLAY_NAMES.get(d["label"], d["label"]) + " (model unclear)"


TARGET, CONTEXT = load()
TEK_PATH = os.environ.get("TEK_CLASSIFIER", os.path.join(ROOT, "models", "deploy", "tek_classifier.pth"))
TEK_MIN_CONF = float(os.environ.get("TEK_MIN_CONF", "0.6"))
# LABEL_MODE=generic (default): every target is answered as "oscilloscope" - the reliable part today
# (86-89% found on real test photos vs ~60% with the exact model). The model guess stays in "model_hint".
# LABEL_MODE=models: answer with the exact model (rs_rtb2004 / tek_tds2014 / tek_tds1002).
LABEL_MODE = os.environ.get("LABEL_MODE", "generic").lower()
TEK = TekClassifier(TEK_PATH) if TARGET and os.path.exists(TEK_PATH) and     os.environ.get("TEK_CHECK", "on").lower() not in ("off", "0", "false", "no") else None
app = FastAPI(title="Leonardo AR - oscilloscope finder")


@app.middleware("http")
async def password_gate(request: Request, call_next):
    """Optional HTTP basic auth for hosted demos (health stays open for the platform's checks)."""
    if DEMO_PASSWORD and request.url.path != "/health":
        ok = False
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("basic "):
            try:
                _, _, pw = base64.b64decode(auth[6:]).decode().partition(":")
                ok = secrets.compare_digest(pw, DEMO_PASSWORD)
            except ValueError:
                ok = False
        if not ok:
            return Response("password required", 401, {"WWW-Authenticate": 'Basic realm="oscilloscope finder"'})
    return await call_next(request)


@app.get("/health")
def health():
    return {
        "status": "ok" if TARGET else "degraded",
        "recognition_mode": "real",
        "inference": f"yolox_tiny_{TARGET.prefix}" if TARGET else "unavailable",
        "label_mode": LABEL_MODE,
        "classes": ([{"label": "oscilloscope", "display_name": "Oscilloscope", "colour": "#18c27f"}]
                    if LABEL_MODE == "generic" else
                    [{"label": n, "display_name": DISPLAY_NAMES.get(n, n), "colour": COLOURS.get(n, "#18c27f")}
                     for n in TARGET.names]) if TARGET else [],
        "context_inference": "yolox_tiny_coco" if CONTEXT else None,
        "device": "cpu",
        "tiled_detection": TILES,
        "tektronix_check": bool(TEK),
        "capture": CAPTURE,
        "min_confidence": TARGET_MIN_CONF,
        # Render sets RENDER=true; the page then says photos go to the demo server instead of "runs locally"
        "hosted": bool(os.environ.get("RENDER")),
        "target_checkpoint": os.path.relpath(TARGET.ckpt, ROOT) if TARGET else None,
    }


async def _read_image(image):
    raw = await image.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, f"image larger than {MAX_UPLOAD // (1024 * 1024)} MB")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "image could not be decoded (use JPEG or PNG)")
    return img


def _capture(img, reason, detections, note=""):
    """Store a frame for labelling: raw/captures/<date>/<time>_<reason>.jpg + .json (model boxes, reason)."""
    day = datetime.date.today().isoformat()
    folder = os.path.join(CAPTURE_ROOT, day)
    os.makedirs(folder, exist_ok=True)
    stem = datetime.datetime.now().strftime("%H%M%S_%f")[:-3] + "_" + reason
    cv2.imwrite(os.path.join(folder, stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(folder, stem + ".json"), "w") as f:
        json.dump({"reason": reason, "note": note[:500], "detections": detections}, f, indent=1)
    return os.path.relpath(os.path.join(folder, stem + ".jpg"), ROOT).replace(os.sep, "/")


@app.post("/v1/recognitions")
async def recognitions(image: UploadFile = File(...), session_id: str = Form(None),
                       frame_timestamp_ms: int = Form(None), capture: bool = Form(False)):
    if TARGET is None:
        raise HTTPException(503, "oscilloscope model weights are not available yet")
    img = await _read_image(image)
    t0 = time.perf_counter()
    # detect down to the "unsure" level, answer only with confident boxes
    candidates = TARGET.detect(img, min(TARGET_MIN_CONF, UNSURE_RANGE[0]), TILES)
    targets = [d for d in candidates if d["confidence"] >= TARGET_MIN_CONF]
    apply_tek_check(img, targets)          # better model names (also used for the hint in generic mode)
    if LABEL_MODE == "generic":
        for d in targets:
            d["model_hint"] = {"label": d["label"], "display_name": d["display_name"]}
            d["label"], d["display_name"], d["class_id"] = "oscilloscope", "Oscilloscope", "target:oscilloscope"
    for d in targets:
        d["target"] = True
    captured = None
    # learning from real use: only if the server has capture on AND this client opted in (capture=true)
    if CAPTURE and capture and any(UNSURE_RANGE[0] <= d["confidence"] < UNSURE_RANGE[1] for d in candidates):
        captured = _capture(img, "unsure", candidates)
    context = []
    if CONTEXT:
        for d in CONTEXT.detect(img, COCO_MIN_CONF, TILES and CONTEXT_TILES):
            # the generic model often calls an oscilloscope "tv", "microwave" or "laptop": the specific model wins
            if any(_covered(d["bbox"], t["bbox"]) > 0.5 for t in targets):
                continue
            context.append(dict(d, target=False))
    return {
        "request_id": str(uuid.uuid4()),
        "session_id": session_id,
        "frame_timestamp_ms": frame_timestamp_ms,
        "mode": "real",
        "simulated": False,
        "inference": f"yolox_tiny_{TARGET.prefix}" + ("+coco" if CONTEXT else ""),
        "device": "cpu",
        "image": {"width": img.shape[1], "height": img.shape[0]},
        "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
        "detections": sorted(targets, key=lambda d: -d["confidence"]) + sorted(context, key=lambda d: -d["confidence"]),
        "captured": bool(captured),
    }


@app.post("/v1/feedback")
async def feedback(image: UploadFile = File(...), note: str = Form("")):
    """"This result is wrong": store the photo (and the model's boxes) for labelling. Lab server only."""
    if not CAPTURE:
        raise HTTPException(403, "feedback capture is off on this server (CAPTURE_MODE=off or hosted demo)")
    img = await _read_image(image)
    candidates = TARGET.detect(img, UNSURE_RANGE[0], TILES) if TARGET else []
    return {"stored": _capture(img, "reported", candidates, note)}


@app.get("/samples")
def samples():
    folder = os.path.join(ROOT, "web", "samples")
    return sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))) if os.path.isdir(folder) else []


app.mount("/samples", StaticFiles(directory=os.path.join(ROOT, "web", "samples"), check_dir=False), name="samples")


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "web", "index.html"))
