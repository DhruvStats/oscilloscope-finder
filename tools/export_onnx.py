"""Export the deployed detector to ONNX for edge / mobile / headset use, and check it matches PyTorch.

    .venv/Scripts/python tools/export_onnx.py [--ckpt models/deploy/yolox_tiny_osc3.pth]

Writes models/export/yolox_tiny_osc3.onnx and a .json next to it describing input and output:
  input  "images": float32 [1, 3, 416, 416], BGR, letterboxed (resize keeping aspect, pad with 114, top-left),
                   pixel values 0-255, no normalisation
  output "output": float32 [1, N, 5 + classes] per candidate: cx, cy, w, h (input pixels), objectness,
                   class probabilities. Score = objectness * class probability; then NMS (IoU 0.65) and a
                   confidence cut-off (0.4 in the app).
The ONNX file runs in Unity Sentis (Quest), ONNX Runtime (PC/phone), OpenVINO (Intel) and TensorRT (NVIDIA).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models", "yolox"))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from yolox.data.data_augment import ValTransform  # noqa: E402

from config import registry  # noqa: E402
from yolox_tiny_osc3 import Exp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "deploy", "yolox_tiny_osc3.pth"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "export", "yolox_tiny_osc3.onnx"))
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    exp = Exp()
    model = exp.get_model().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.head.decode_in_inference = True          # output decoded boxes, ready for NMS
    h, w = exp.test_size
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 3, h, w), args.out, input_names=["images"], output_names=["output"],
                      opset_version=args.opset, dynamo=False)

    # check: same output as PyTorch on a real photo
    import onnxruntime as ort
    img = cv2.imread(os.path.join(ROOT, "raw", "photos", "61.jpg"))
    x, _ = ValTransform(legacy=False)(img, None, (h, w))
    x = x[None].astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"images": x})[0]
    diff = float(np.abs(ref - got).max())
    names = registry.labels()

    def top(out):
        s = out[0, :, 4:5] * out[0, :, 5:]
        i = np.unravel_index(np.argsort(s, axis=None)[::-1][:3], s.shape)
        return [(names[c], round(float(s[r, c]), 3)) for r, c in zip(*i)]

    meta = {
        "model": "YOLOX-Tiny", "classes": names, "input": {"name": "images", "shape": [1, 3, h, w],
                                                           "format": "BGR, letterbox pad 114, 0-255"},
        "output": {"name": "output", "layout": "cx, cy, w, h, objectness, class probabilities"},
        "postprocess": {"score": "objectness * class_probability", "nms_iou": exp.nmsthre, "confidence": 0.4},
        "source_checkpoint": os.path.relpath(args.ckpt, ROOT).replace(os.sep, "/"),
        "check": {"max_abs_diff_vs_pytorch": diff},
    }
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"written {os.path.relpath(args.out, ROOT)} ({os.path.getsize(args.out) / 2**20:.1f} MB)")
    print(f"max difference vs PyTorch: {diff:.2e}")
    print("top scores  PyTorch:", top(ref))
    print("            ONNX   :", top(got))
    if diff > 1e-2:
        sys.exit("ONNX output differs from PyTorch - do not use this file")


if __name__ == "__main__":
    main()
