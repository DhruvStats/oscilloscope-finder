# New object kit - reuse this pipeline for PCBs, coffee machines, anything

The oscilloscope project is one instance of a general pipeline. For a new object (e.g. a PCB board or a coffee
machine) nothing in the code is specific to oscilloscopes except `config/instruments.yaml`.

## 1. Register the object (5 min)
`config/instruments.yaml` - replace or add entries (the order is the model's class order):
```yaml
instruments:
  - label: coffee_machine_x        # lowercase_with_underscores
    display_name: Coffee machine X
    colour: "#c0392b"
    width_mm: 300                  # real width: keeps relative sizes right in generated scenes
    sample_weight: 1.0             # >1 for objects that are hard to recognise
    hints: what tells it apart from similar objects
    textures: []                   # optional 3D box textures; without them real cut-outs are used
```
Then `python tools/registry_sync.py` (updates the Label Studio interface).

## 2. Collect real data (the part that decides accuracy)
- **Photos:** 30-60 per object, all sides, near and far, several rooms and lights, with normal clutter.
- **Videos:** 1-2 min walk-arounds per object -> `python tools/extract_frames.py video.mp4 raw/<session>/frames --count 200`.
- **The real use case:** frames taken the way the app will be used (e.g. Quest camera, user in front of the object).
- **Negatives:** look-alikes that are NOT the object (for a PCB: other boards, laptops, keyboards; for a coffee
  machine: microwaves, printers, water dispensers).
- Keep one session (a room / a day) completely apart as the **test set**.

## 3. Label (model-assisted)
- First round: draw boxes in Label Studio (`labelstudio/start.ps1`, `make_tasks.py --new raw/<folder> --only-new`,
  `add_tasks.py`), or review proposals of an existing model with `tools/review_batch.py` + `tools/review_crops.py`.
- Export JSON -> `labelstudio/import_export.py` -> `raw/real_labels.json`.
- **Box every visible instance**, even partly hidden - an unboxed object teaches "this is not the object".

## 4. Build the dataset
```powershell
.venv\Scripts\python synth\gen3.py --per-class 300 --real-labels raw\real_labels.json
.venv\Scripts\python tools\export_formats.py        # COCO + YOLO + Pascal VOC
```
Generated scenes (real cut-outs on new backgrounds, crops, look-alikes pasted unlabelled) + real images.
80/20 train/validation of the generated scenes, plus ~12% real images held out for choosing the model, plus the
real test session.

## 5. Train, mine mistakes, retrain
```powershell
.venv\Scripts\python training\train_cpu.py --exp yolox_tiny_osc3 --epochs 30 --init models\yolox_tiny.pth
.venv\Scripts\python tools\mine_false_positives.py      # what does it wrongly detect in the train images?
.venv\Scripts\python tools\add_mined_labels.py          # missed objects -> labels, real false alarms -> negatives
# rebuild the dataset (step 4) and train again from the last model
```
On the NVIDIA workstation: `tools\package_for_workstation.py` (minutes instead of hours).

## 6. Test honestly and deploy
```powershell
.venv\Scripts\python tools\eval_app.py models\training\yolox_tiny_osc3\best_ckpt.pth --generic
```
Shows overall accuracy and the **use-case** accuracy (object >= 3% of the image). Deploy only if it beats the
current model: copy to `models\deploy\`, commit, push -> Render redeploys; `tools\export_onnx.py` for Unity Sentis.

## What changes per object
| Object type | Watch out for |
|---|---|
| Very similar models (like TDS 2014 vs 1002) | add a second-stage classifier on crops (`training/train_tek_classifier.py` as template) or answer the generic class |
| Flat objects (PCB) | many angles incl. top-down; similar boards as negatives; small text is not readable at 416 px - use crops |
| Large appliances (coffee machine) | partial views (only the front panel), different models of the same brand as negatives |
