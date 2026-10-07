# Oscilloscope Detection: Brand and Model Number Recognition

Given any image, the system finds each oscilloscope, identifies its brand, and reads the
model number printed on the front panel.

```
image -> YOLOv8 detector -> box + brand class -> crop -> EasyOCR -> fuzzy match to known models -> result .txt
```

Currently trained for: **Rohde & Schwarz RTB2004**. Two more brands will be added.

## Project layout

| Path | Purpose |
|---|---|
| `configs/oscilloscopes.yaml` | List of oscilloscope models (brand, model number, dataset folder) |
| `scripts/prepare_dataset.py` | Converts COCO annotations to YOLO `.txt` labels, makes the 80/20 split |
| `scripts/train.py` | Trains YOLOv8, saves `models/best.pt` |
| `scripts/evaluate.py` | Measures accuracy on the real test photos |
| `scripts/predict.py` | Runs the full pipeline on an image or folder, writes result `.jpg` + `.txt` |
| `oscilloscope_pipeline.py` | Detection + OCR + model-number matching logic |
| `app.py` | Streamlit web frontend |

## How to run

```bash
pip install -r requirements.txt
python scripts/prepare_dataset.py
python scripts/train.py
python scripts/evaluate.py
python scripts/predict.py data/yolo/images/test
streamlit run app.py
```

## Dataset and split

Source: `Downloads/dataset-rtb2004` (COCO format).

| Split | Images | Content |
|---|---|---|
| train | 143 | synthetic (80%) |
| val | 36 | synthetic (20%) |
| test | 8 | **real phone photos**, never used in training |

- The synthetic images are 3D renders of the RTB2004 on cluttered backgrounds, so their boxes are pixel-exact.
- About 15% contain no oscilloscope. These teach the model not to fire on clutter, and get an empty `.txt` label.
- The real photos are kept as a separate test set, so the reported accuracy reflects real-world use.
- During training, YOLO also augments images on the fly (mosaic, scaling, brightness/colour).
  Horizontal flipping is disabled because it would mirror the logo and model text.

## The two kinds of text files

**1. Label files** (`data/yolo/labels/*/*.txt`): the training answers, one per image:
```
<class_id> <x_center> <y_center> <width> <height>     # all values 0-1, relative to image size
0 0.434375 0.438542 0.659375 0.877083
```
`classes.txt` maps class IDs to names (`0 = rs_rtb2004`).

**2. Result files** (`outputs/*_result.txt`, or the app's download button): the system's output:
```
image: real_01.jpg
oscilloscopes_found: 1

[oscilloscope 1]
brand: Rohde & Schwarz
model_number: RTB2004
model_number_source: OCR
detection_confidence: 0.91
ocr_match_score: 100
ocr_raw_text: R&S RTB2004 | Digital Oscilloscope | ...
bbox_xyxy: 101 463 516 1095
```
`model_number_source` is `OCR` when the text was read from the panel, or `detector` when the
text was unreadable and the brand/model came from the detector class alone.

## Training on a free GPU (Google Colab)

Training on this laptop's CPU takes about 10 minutes per epoch. On Colab it takes seconds per epoch.

1. `python scripts/prepare_dataset.py` (re-run whenever the data changes), then zip it:
   `python -c "import shutil; shutil.make_archive('colab_dataset','zip','data','yolo')"`
2. Open `train_colab.ipynb` at https://colab.research.google.com (File -> Upload notebook), select a T4 GPU, and run all cells.
3. Put the downloaded `best.pt` into `models/`. Then `evaluate.py`, `predict.py`, and the app use it.

## Current status (first experiment)

| Item | Result |
|---|---|
| Synthetic validation mAP@0.5 | ~0.96 after 1 epoch |
| **Real test photos** mAP@0.5 | **0.07** (checkpoint after only 9 CPU epochs) |
| OCR on a front-view crop | reads `RTB2004` exactly (match score 100) |

Finding: a model trained only on synthetic renders transfers poorly to real photos (the
"synthetic-to-real gap"). The fixes are (1) full training on GPU and (2) adding real photos:
take 50-100 photos of the oscilloscope from all angles, label them in Label Studio, export as
COCO, and list the folder under `real_photos:` in the config.

## Adding real photos labeled in Label Studio

1. Create a project with the *Object Detection with Bounding Boxes* template, using the label `rtb2004`.
2. Import the photos and draw one box per oscilloscope.
3. Export -> **COCO** -> unzip (you get `result.json` + `images/`).
4. Add the folder to `real_photos:` in `configs/oscilloscopes.yaml` and re-run `prepare_dataset.py`.

## Adding a new brand

1. Get a dataset for it in the same COCO layout (`images/{train,val,test}2017`,
   `annotations/instances_{train,val,test}2017.json`). Alternatively, label photos in Label Studio
   and export as COCO.
2. Add an entry to `configs/oscilloscopes.yaml` (name, brand, model, source folder, COCO category name).
3. Re-run `prepare_dataset.py` and `train.py`. The OCR matcher picks up the new model number automatically.
