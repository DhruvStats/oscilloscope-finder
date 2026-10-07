"""Detect oscilloscopes in an image, identify brand, and read the model number.

Steps for every image:
    1. YOLO finds each oscilloscope -> bounding box + class (class gives the brand/model).
    2. The box is cropped, enlarged and contrast-enhanced.
    3. EasyOCR reads all text on the front panel.
    4. The OCR text is fuzzy-matched against the known model numbers from
       configs/oscilloscopes.yaml, which fixes typical OCR mistakes (0/O, 1/I, 5/S ...).
    5. A result .txt is produced (see to_text).
"""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from rapidfuzz import fuzz
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "models" / "best.pt"
DEFAULT_CONFIG = ROOT / "configs" / "oscilloscopes.yaml"

OCR_MATCH_THRESHOLD = 80  # 0-100 similarity needed to accept an OCR reading as a model number


@dataclass
class Detection:
    box: tuple            # x1, y1, x2, y2 in pixels
    class_name: str
    brand: str
    detector_model: str   # model implied by the YOLO class
    confidence: float
    ocr_text: str = ""
    model_number: str = ""
    ocr_score: float = 0.0
    model_source: str = ""  # "OCR" or "detector"


@dataclass
class Result:
    image_name: str
    detections: list = field(default_factory=list)


def _normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


class OscilloscopeAnalyzer:
    def __init__(self, weights=DEFAULT_WEIGHTS, config=DEFAULT_CONFIG, use_ocr=True):
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"{weights} not found - train the model first (scripts/train.py)")
        self.classes = {c["name"]: c for c in yaml.safe_load(Path(config).read_text())["classes"]}
        self.known_models = [c["model"] for c in self.classes.values()]
        self.use_ocr = use_ocr
        self._reader = None
        # On Windows, PyTorch crashes or hangs when its models are used from more than one thread
        # (the Streamlit app runs every re-run on a new thread). So all model loading and inference
        # happens on this one long-lived worker thread; analyze() hands work to it and waits.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oscilloscope-model")
        self.model = self._worker.submit(YOLO, str(weights)).result()

    @property
    def reader(self):
        # EasyOCR is slow to load (and downloads its weights on first use), so load it lazily.
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return self._reader

    def analyze(self, image: np.ndarray, image_name="image", conf=0.25) -> Result:
        """image is a BGR numpy array (as returned by cv2.imread)."""
        return self._worker.submit(self._analyze, image, image_name, conf).result()

    def _analyze(self, image, image_name, conf):
        result = Result(image_name)
        pred = self.model.predict(image, conf=conf, verbose=False)[0]
        for xyxy, cls, score in zip(pred.boxes.xyxy.tolist(), pred.boxes.cls.tolist(), pred.boxes.conf.tolist()):
            name = pred.names[int(cls)]
            info = self.classes.get(name, {"brand": "unknown", "model": "unknown"})
            det = Detection(tuple(int(v) for v in xyxy), name, info["brand"], info["model"], float(score))
            if self.use_ocr:
                self._read_model_number(image, det)
            else:
                det.model_number, det.model_source = det.detector_model, "detector"
            result.detections.append(det)
        result.detections.sort(key=lambda d: d.confidence, reverse=True)
        return result

    def _read_model_number(self, image, det: Detection):
        crop = self._prepare_crop(image, det.box)
        texts = self.reader.readtext(crop, detail=0, paragraph=False)
        best_model, best_score = self._match_model(texts)
        if best_score < OCR_MATCH_THRESHOLD:
            # Phone photos are often sideways: retry with the crop rotated 90/180/270 degrees.
            texts = self.reader.readtext(crop, detail=0, paragraph=False, rotation_info=[90, 180, 270])
            rotated_model, rotated_score = self._match_model(texts)
            if rotated_score > best_score:
                best_model, best_score = rotated_model, rotated_score
        det.ocr_text = " | ".join(texts)

        det.ocr_score = best_score
        if best_model and best_score >= OCR_MATCH_THRESHOLD:
            det.model_number, det.model_source = best_model, "OCR"
        else:
            # Text not readable (side/back view, too small, blurred): fall back to the detector's class.
            det.model_number, det.model_source = det.detector_model, "detector"

    def _match_model(self, texts):
        """Return (known model, similarity 0-100) that best matches the OCR text."""
        candidates = [_normalize(t) for t in texts] + [_normalize("".join(texts))]
        best_model, best_score = None, 0.0
        for model in self.known_models:
            m = _normalize(model)
            for c in candidates:
                if len(c) < 0.7 * len(m):  # too short to be this model number
                    continue
                score = fuzz.partial_ratio(m, c)
                if score > best_score:
                    best_model, best_score = model, score
        return best_model, best_score

    @staticmethod
    def _prepare_crop(image, box, pad=0.05, min_width=1000):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        crop = image[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
        if crop.shape[1] < min_width:
            scale = min_width / crop.shape[1]
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def annotate(image: np.ndarray, result: Result) -> np.ndarray:
    out = image.copy()
    thickness = max(2, image.shape[1] // 400)
    font_scale = max(0.6, image.shape[1] / 1200)
    for d in result.detections:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), thickness)
        label = f"{d.brand} {d.model_number} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(th + 8, y1)
        cv2.rectangle(out, (x1, ty - th - 8), (x1 + tw + 8, ty), (0, 200, 0), -1)
        cv2.putText(out, label, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    return out


def to_text(result: Result) -> str:
    lines = [f"image: {result.image_name}", f"oscilloscopes_found: {len(result.detections)}"]
    for i, d in enumerate(result.detections, 1):
        lines += [
            "",
            f"[oscilloscope {i}]",
            f"brand: {d.brand}",
            f"model_number: {d.model_number}",
            f"model_number_source: {d.model_source}",
            f"detection_confidence: {d.confidence:.2f}",
            f"ocr_match_score: {d.ocr_score:.0f}",
            f"ocr_raw_text: {d.ocr_text}",
            f"bbox_xyxy: {d.box[0]} {d.box[1]} {d.box[2]} {d.box[3]}",
        ]
    return "\n".join(lines) + "\n"
