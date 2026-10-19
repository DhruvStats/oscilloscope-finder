"""Figures for docs/build_report.py (writes docs/img/report_*.png|jpg).

    .venv/Scripts/python docs/make_report_figures.py
"""
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "docs", "img")
sys.path.insert(0, ROOT)
os.environ["CONTEXT_MODEL"] = "off"
import server.app as A  # noqa: E402

COL = {"rs_rtb2004": (127, 194, 24), "tek_tds2014": (31, 138, 255), "tek_tds1002": (209, 75, 209)}   # BGR
NAME = {"rs_rtb2004": "R&S RTB2004", "tek_tds2014": "Tektronix TDS 2014", "tek_tds1002": "Tektronix TDS 1002"}


def run(path):
    img = cv2.imread(path)
    dets = [d for d in A.TARGET.detect(img, A.TARGET_MIN_CONF, True) if d["confidence"] >= A.TARGET_MIN_CONF]
    A.apply_tek_check(img, dets)
    A.apply_detector_confirm(dets)
    return img, dets


def draw(img, dets, truth=None, width=1400):
    img = img.copy()
    H, W = img.shape[:2]
    lw, fs = max(3, W // 300), max(0.9, W / 1300)
    for d in dets:
        b = d["bbox"]
        x, y, w, h = int(b["x"] * W), int(b["y"] * H), int(b["width"] * W), int(b["height"] * H)
        c = COL[d["label"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), c, lw)
        text = f'{NAME[d["label"]]} {d["confidence"]:.0%}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        ty = max(th + 10, y)
        x = min(x, W - tw - 12)          # keep the label inside the image
        cv2.rectangle(img, (x - lw // 2, ty - th - 10), (x + tw + 10, ty), c, -1)
        cv2.putText(img, text, (x + 5, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
    s = width / W
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def main():
    os.makedirs(IMG, exist_ok=True)
    # 1. a scene with all three models (training photo, for illustration)
    img, dets = run(os.path.join(ROOT, "raw", "batch_2026-10-08", "photos", "06.jpg"))
    cv2.imwrite(os.path.join(IMG, "report_example.jpg"), draw(img, dets), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("example:", [(d["label"], round(d["confidence"], 2)) for d in dets])

    # 2. the typical remaining error (held-out test frame, unseen lab): TDS 2014 from above called TDS 1002
    p = os.path.join(ROOT, "datasets", "oscilloscopes3", "test2017", "real_frames_v00_v00_000188.jpg")
    img, dets = run(p)
    out = draw(img, dets, width=700)
    out = out[int(out.shape[0] * 0.30):int(out.shape[0] * 0.82)]      # the bench part of the portrait frame
    cv2.imwrite(os.path.join(IMG, "report_error.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("error case:", [(d["label"], round(d["confidence"], 2)) for d in dets])

    # 3. what the three models look like (real crops)
    picks = {"rs_rtb2004": "raw/photos/02.jpg", "tek_tds2014": "raw/photos/24.jpg", "tek_tds1002": "raw/photos/15.jpg"}
    import json
    labels = {i["file"]: i["boxes"] for i in json.load(open(os.path.join(ROOT, "raw", "real_labels.json")))["images"]}
    tiles = []
    for cls, f in picks.items():
        im = cv2.imread(os.path.join(ROOT, f))
        x, y, w, h = next(b["bbox"] for b in labels[f] if b["cls"] == cls)
        cr = im[max(0, y - 10):y + h + 10, max(0, x - 10):x + w + 10]
        cr = cv2.resize(cr, (600, int(600 * cr.shape[0] / cr.shape[1])))
        cr = cv2.copyMakeBorder(cr, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
        bar = np.full((60, 600, 3), COL[cls], np.uint8)
        cv2.putText(bar, NAME[cls], (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(np.vstack([bar, cr]))
    Hm = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, Hm - t.shape[0], 0, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    cv2.imwrite(os.path.join(IMG, "report_models.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 4. results chart (numbers from tools/eval_app.py on the 53-image real test)
    versions = ["v5", "v6", "v7", "v8", "v9", "deployed\n(avg v7-v9)"]
    use_models = [41, 42, 43, 41, 41, 44]
    all_models = [61, 60, 67, 58, 64, 68]
    use_generic = [48, 50, 50, 49, 49, 50]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), dpi=160)
    x = np.arange(len(versions))
    ax[0].bar(x - 0.2, [v / 50 * 100 for v in use_generic], 0.4, label='"oscilloscope or not"', color="#9fd8c0")
    ax[0].bar(x + 0.2, [v / 50 * 100 for v in use_models], 0.4, label="exact model", color="#0f6e56")
    ax[0].set_title("Use case: scope >= 3% of the image (50 scopes)", fontsize=10)
    ax[1].bar(x, [v / 85 * 100 for v in all_models], 0.55, color="#0f6e56")
    ax[1].set_title("Exact model, all 85 scopes (incl. tiny, far away)", fontsize=10)
    for a in ax:
        a.set_xticks(x, versions, fontsize=8)
        a.set_ylim(50, 102)
        a.set_ylabel("% correct", fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=.3)
    ax[0].legend(fontsize=8, loc="lower right", frameon=False)
    for i, v in enumerate(use_models):
        ax[0].text(i + 0.2, v / 50 * 100 + 0.8, f"{v / 50:.0%}", ha="center", fontsize=7)
    for i, v in enumerate(all_models):
        ax[1].text(i, v / 85 * 100 + 0.8, f"{v}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "report_results.png"))
    print("figures written to", IMG)


if __name__ == "__main__":
    main()
