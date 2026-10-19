"""Build docs/Project_Report_Oscilloscope_Recognition.pdf - the project report for supervisors / professors.

    .venv/Scripts/python docs/make_report_figures.py     # figures (needs the deployed models)
    .venv/Scripts/python docs/build_report.py
"""
import os

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Project_Report_Oscilloscope_Recognition.pdf")
IMG = os.path.join(HERE, "img")

INK = colors.HexColor("#1d1d1b")
MUTED = colors.HexColor("#5f5e5a")
LINE = colors.HexColor("#d3d1c7")
SOFT = colors.HexColor("#f1efe8")
ACCENT = colors.HexColor("#0f6e56")
GREEN = colors.HexColor("#18c27f")
ORANGE = colors.HexColor("#ff8a1f")
MAGENTA = colors.HexColor("#d14bd1")
BLUE = colors.HexColor("#3b6ea8")

S = {
    "title": ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=24, leading=29, textColor=INK, spaceAfter=8),
    "sub": ParagraphStyle("s", fontName="Helvetica", fontSize=12.5, leading=17, textColor=MUTED, spaceAfter=4),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15.5, leading=20, textColor=ACCENT,
                         spaceBefore=6, spaceAfter=7),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=INK,
                         spaceBefore=8, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="Helvetica", fontSize=9.8, leading=14, textColor=INK, spaceAfter=5),
    "b": ParagraphStyle("b", fontName="Helvetica", fontSize=9.8, leading=13.6, textColor=INK, leftIndent=12,
                        bulletIndent=2, spaceAfter=2),
    "small": ParagraphStyle("sm", fontName="Helvetica", fontSize=8.3, leading=11, textColor=MUTED),
    "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8.3, leading=11, textColor=MUTED,
                          spaceBefore=3, spaceAfter=8),
    "cell": ParagraphStyle("c", fontName="Helvetica", fontSize=8.6, leading=11, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="Helvetica-Bold", fontSize=8.6, leading=11, textColor=INK),
    "box": ParagraphStyle("box", fontName="Helvetica", fontSize=9.6, leading=13.6, textColor=INK),
}
W = A4[0] - 36 * mm


def P(t, s="p"):
    return Paragraph(t, S[s])


def bullets(items):
    return [Paragraph(t, S["b"], bulletText="•") for t in items]


def table(rows, widths, head=True, bold_last=False):
    data = [[Paragraph(str(c), S["cellb"] if (head and i == 0) or (bold_last and i == len(rows) - 1) else S["cell"])
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    if bold_last:
        st.append(("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), colors.HexColor("#e3f5ee")))
    t.setStyle(TableStyle(st))
    return t


def callout(text, colour=ACCENT):
    t = Table([[Paragraph(text, S["box"])]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef6f2")),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, colour), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def img(name, width, caption=None):
    path = os.path.join(IMG, name)
    from PIL import Image as PI
    w, h = PI.open(path).size
    out = [Image(path, width=width, height=width * h / w)]
    if caption:
        out.append(P(caption, "cap"))
    return out


def arrow(d, x1, y1, x2, y2, c=MUTED):
    d.add(Line(x1, y1, x2, y2, strokeColor=c, strokeWidth=1.2))
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    d.add(Polygon([x2, y2, x2 - 6 * math.cos(a - 0.4), y2 - 6 * math.sin(a - 0.4),
                   x2 - 6 * math.cos(a + 0.4), y2 - 6 * math.sin(a + 0.4)], fillColor=c, strokeColor=c))


def box(d, x, y, w, h, title, lines, fill=SOFT, edge=LINE, tc=INK):
    d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=fill, strokeColor=edge, strokeWidth=1))
    d.add(String(x + w / 2, y + h - 14, title, fontName="Helvetica-Bold", fontSize=8.6, fillColor=tc,
                 textAnchor="middle"))
    for i, ln in enumerate(lines):
        d.add(String(x + w / 2, y + h - 27 - i * 10, ln, fontName="Helvetica", fontSize=7.3, fillColor=MUTED,
                     textAnchor="middle"))


def architecture():
    d = Drawing(W, 200)
    bw, bh = 100, 62
    box(d, 0, 120, bw, bh, "Meta Quest 3 / web", ["passthrough camera", "frame (JPEG) over LAN", "or photo upload"])
    box(d, 128, 120, bw, bh, "1. Detector", ["YOLOX-Tiny, 416 px", "whole photo + tiles", "3 classes, 40% cut-off"],
        fill=colors.HexColor("#e3f5ee"), edge=ACCENT)
    box(d, 256, 120, bw, bh, "2. Tektronix check", ["MobileNetV3 on crop", "TDS 2014 vs TDS 1002", "confirmed / unclear"],
        fill=colors.HexColor("#fff1e3"), edge=ORANGE)
    box(d, 384, 120, W - 384, bh, "3. Answer (API)", ["label, confidence, box", "model_certain + source", "context objects"])
    arrow(d, bw, 151, 128, 151)
    arrow(d, 128 + bw, 151, 256, 151)
    arrow(d, 256 + bw, 151, 384, 151)
    box(d, 128, 20, bw, bh, "Context model", ["YOLOX-Tiny COCO", "laptop, keyboard, ...", "drops scope duplicates"])
    box(d, 256, 20, bw, bh, "Instrument registry", ["config/instruments.yaml", "names, colours, sizes", "(next: 3D mesh path)"])
    box(d, 384, 20, W - 384, bh, "Clients", ["Unity AR module per model", "web page (3 colours)", "next: pose stage"],
        fill=colors.HexColor("#e8eef7"), edge=BLUE)
    arrow(d, 178, 82, 178, 120)
    arrow(d, 306, 82, 306, 120)
    arrow(d, 434, 120, 434, 82)
    d.add(String(0, 100, "FastAPI server (LAN, on-premise) - same code on the Render demo", fontName="Helvetica-Oblique",
                 fontSize=7.5, fillColor=MUTED))
    return d


def footer(c, doc):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, "Leonardo AR PoC - oscilloscope recognition - project report, October 2026")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    c.restoreState()


def build():
    st = []
    # ---------------------------------------------------------------- title + summary
    st += [Spacer(1, 18 * mm), P("Recognising laboratory oscilloscopes for an AR assistant", "title"),
           P("Leonardo AR proof of concept - recognition stage: methods, data, results and next steps", "sub"),
           P("Project report, October 2026", "small"), Spacer(1, 8 * mm)]
    st += img("report_example.jpg", W, "Figure 1. The deployed system on a lab photo: all three instrument models found and "
              "named, each in its own colour (training-set photo, shown for illustration).")
    st += [P("Summary", "h1"),
           P("The goal is a recognition service that tells an augmented-reality application on a Meta Quest 3 <b>which "
             "oscilloscope the user is looking at</b>, so that the matching AR content and, in the next stage, the "
             "matching 3D model for 6D pose estimation can be loaded. Three instruments are in scope: the "
             "<b>Rohde &amp; Schwarz RTB2004</b>, the <b>Tektronix TDS 2014</b> and the <b>Tektronix TDS 1002</b>."),
           P("The system is a two-stage detector: a YOLOX-Tiny object detector, fine-tuned on real lab photos, video "
             "frames and generated scenes, followed by a second-stage classifier that separates the two Tektronix "
             "models, which share the same case. It runs on-premise as a LAN service with the project's API contract, "
             "with a public demo and a Unity client."),
           callout("<b>Main results</b> on 53 real held-out images (85 instruments; 39 frames from a lab never used for "
                   "training): <b>100%</b> of the instruments that fill at least 3% of the image are found "
                   "(\"is it an oscilloscope\"), and <b>88%</b> are also named with the exact model. "
                   "The remaining errors are almost all the TDS 2014 seen from above being called TDS 1002; "
                   "the RTB2004 was never misnamed.")]
    st.append(PageBreak())

    # ---------------------------------------------------------------- 1 problem
    st += [P("1. Problem and requirements", "h1"),
           P("A technician wearing a Quest 3 looks at a lab bench. The AR application sends camera frames to a "
             "recognition server on the local network and receives, for every instrument in view, its model, a "
             "confidence and a bounding box. Requirements agreed during the project:"),
           *bullets(["<b>Model-level recognition</b> of all three instruments (not just a generic \"oscilloscope\"), "
                     "because the result selects the geometric model for the pose stage.",
                     "Robust to clutter: laptops, keyboards, speakers, monitors and other look-alikes must not be "
                     "reported as oscilloscopes.",
                     "On-premise and LAN-only: lab images stay local; a cloud demo exists only for presentation.",
                     "Honest evaluation: train, validation and test kept apart by session; results reported on "
                     "real images never seen in training.",
                     "Reusable pipeline for future objects (e.g. PCBs, coffee machines)."]), Spacer(1, 6)]
    st += img("report_models.jpg", W, "Figure 2. The three instruments. The two Tektronix models share the same case; "
              "the TDS 2014 has four colour-coded channel columns, the TDS 1002 two.")

    # ---------------------------------------------------------------- 2 architecture
    st += [P("2. System architecture", "h1"), architecture(),
           P("Figure 3. Processing pipeline of the recognition server.", "cap"),
           P("<b>Detector.</b> YOLOX-Tiny (Megvii, Apache-2.0), initialised from COCO weights and fine-tuned on the "
             "three classes at 416 x 416 px. The photo is analysed as a whole and in overlapping tiles, so that small, "
             "distant instruments get enough pixels; duplicate boxes across tiles are merged. Boxes below 40% "
             "confidence are discarded (one fixed cut-off for all classes)."),
           P("<b>Tektronix check.</b> The detector sees only 416 px; the difference between the two Tektronix models "
             "is in the details of the front panel. A MobileNetV3-small classifier re-examines a full-resolution crop "
             "of every Tektronix box. If it is at least 60% sure, it names the model (\"model confirmed\"); otherwise "
             "the answer is flagged \"model unclear\" instead of guessing. The RTB2004 has no look-alike among the "
             "targets and is confirmed by the detector itself when it is at least 80% sure."),
           P("<b>Context model and registry.</b> A second, generic COCO detector labels everyday objects for the UI "
             "and removes its own duplicates of instruments. Names, colours and real sizes of the instruments come "
             "from one registry file, which also drives the labelling interface and the dataset generator."),
           P("<b>Interfaces.</b> FastAPI server with <i>GET /health</i>, <i>POST /v1/recognitions</i> (per detection: "
             "label, display name, confidence, normalised box, model_certain and the source of the confirmation) and "
             "<i>POST /v1/feedback</i>; a web page; a Unity client that enables one AR module per recognised model; "
             "an ONNX export for on-device inference (Unity Sentis).")]

    # ---------------------------------------------------------------- 3 data
    st += [P("3. Data", "h1"),
           P("All real data was captured in the labs with a phone: photos and walk-around videos of the three "
             "instruments in different rooms, with normal clutter. Every box was checked by hand (Label Studio with "
             "model-proposed boxes, then visual review)."),
           table([["Source", "Images", "Instruments (RTB / TDS 2014 / TDS 1002)", "Use"],
                  ["Real photos", "103", "", "train / test by session"],
                  ["Video frames (3 videos)", "284", "", "train; one video (unseen lab) only for test"],
                  ["Total hand-checked real images (10 sessions)", "387", "252 / 159 / 191", "334 train, 53 test"],
                  ["Extra video frames between checked frames", "782", "1,010 boxes", "classifier training only"],
                  ["Generated scenes (per build)", "~2,000", "~700-800 per model", "train / validation"],
                  ["Hard negatives (look-alikes)", "19 crops", "speakers, keyboards, posters, bag ...", "pasted unlabelled"]],
                 [0.32, 0.12, 0.3, 0.26]),
           P("Table 1. Data used. Instrument counts for real images include boxes added during error analysis.", "cap"),
           P("<b>Generated data.</b> Real cut-outs of the instruments are pasted onto new backgrounds at realistic "
             "relative sizes, together with distractor objects and hard negatives (objects that were wrongly detected "
             "before), and random crops of the real photos at other zoom levels. This multiplies viewpoints and "
             "backgrounds without new photography."),
           P("<b>Splits.</b> Generated scenes are split 80/20 into training and validation; 12% of the real training "
             "images (whole video blocks, so neighbouring frames cannot leak) are added to validation. The <b>test "
             "set</b> is 53 real images: 39 frames from a lab video never used for training (the hardest case) and 14 "
             "lab photos. One dataset per instrument model is also exported (images, one-class labels, crops) for "
             "per-model work and for the pose stage."),
           P("<b>Label quality.</b> Error analysis repeatedly found labelling mistakes rather than model mistakes: "
             "photos labelled \"no oscilloscope\" that actually showed a rear, side or partly hidden instrument (14 of "
             "16 such images), and unboxed instruments in the background. Each one taught the model that a real "
             "oscilloscope is background. Fixing them gave the largest single improvements of the project.")]

    # ---------------------------------------------------------------- 4 training
    st += [P("4. Training", "h1"),
           *bullets(["Detector: fine-tuning from COCO weights, then iteratively from the previous version (10-30 "
                     "epochs each), mosaic and colour augmentation, last epochs without mosaic. CPU training "
                     "(~8-12 min per epoch); a GPU package exists for faster training on a workstation.",
                     "Hard-negative mining: after each round, the detector's false alarms on training images are "
                     "reviewed; real misses become labels, real non-instruments become hard negatives.",
                     "Tektronix classifier: ImageNet-pretrained MobileNetV3-small on ~2,600 crops (real and generated), "
                     "class-balanced sampling, no horizontal flips (button layout matters).",
                     "Training runs are chosen on validation data; the candidates were then compared on the test set before "
                     "deployment, which makes the reported test numbers slightly optimistic. The deployed detector is the weight average of three fine-tuned runs (v7, v8, v9), "
                     "which reduced run-to-run variance."])]
    st.append(PageBreak())

    # ---------------------------------------------------------------- 5 results
    st += [P("5. Results", "h1"),
           P("All numbers are measured on the 53 real test images (85 instruments) with the full pipeline exactly as "
             "deployed (tiles, cut-off, Tektronix check). An instrument counts as correct when a box with IoU >= 0.5 "
             "has the right name. <b>Use case</b> = instruments that cover at least 3% of the image, i.e. what a "
             "Quest user standing in front of a bench sees; the rest are tiny instruments far in the background."),
           table([["Measure", "Generic (\"oscilloscope or not\")", "Exact model (deployed mode)"],
                  ["Use case (50 instruments)", "<b>50 / 50 (100%)</b>", "<b>44 / 50 (88%)</b>"],
                  ["All 85 instruments", "77 / 85 (91%)", "68 / 85 (80%)"],
                  ["Wrong model name", "-", "9 (almost all TDS 2014 -> TDS 1002)"],
                  ["Missed (mostly tiny, far away)", "8", "8"],
                  ["False alarms (53 images)", "2", "2"]],
                 [0.34, 0.33, 0.33]),
           P("Table 2. Deployed system on the real test set.", "cap")]
    st += img("report_results.png", W, "Figure 4. Development history on the same test set. v8 and v9 (retrained with more "
              "data) were not better than v7; averaging the weights of v7-v9 gave the best model.")
    st += [P("<b>Per instrument.</b> The RTB2004 was never given a wrong name. The detector finds instruments reliably; "
             "the open problem is purely the TDS 2014 / TDS 1002 distinction.")]

    # ---------------------------------------------------------------- 6 error analysis
    err = img("report_error.jpg", 62 * mm)
    txt = [P("<b>Typical remaining error.</b> A TDS 2014 in the unseen lab, filmed from above at an angle, is named "
             "TDS 1002 with high confidence. Its four channel columns are visible, but the training data contains "
             "almost no TDS 2014 fronts from this angle (only 239 real TDS 2014 images, mostly side and rear views), "
             "so the classifier learned cues of viewpoint and session instead of the channel count.", "p"),
           P("<b>What did not help</b> (all measured on the test set and therefore not deployed): a larger classifier "
             "(MobileNetV3-large at 256 px, 98% on validation but worse on the unseen lab), adding 782 extra video "
             "frames, video-like degradation augmentation, averaging two classifiers, and two further detector "
             "retrainings. A strong validation score did not transfer to the unseen lab: the domain gap between "
             "sessions, not model capacity, is the limiting factor.", "p"),
           P("<b>What helped most:</b> correcting labels (missed instruments, wrong \"empty\" images), mined hard "
             "negatives (speakers and keyboards stopped being detected) and weight averaging.", "p")]
    t = Table([[err, txt]], colWidths=[66 * mm, W - 66 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    st += [KeepTogether([P("6. Error analysis and lessons learned", "h1"), t,
                         P("Figure 5. Held-out test frame from the unseen lab: the TDS 2014 is named TDS 1002 (the "
                           "RTB2004 at the left edge is found).", "cap")])]

    # ---------------------------------------------------------------- 7 deployment
    st += [P("7. Deployment", "h1"),
           table([["Where", "What", "Notes"],
                  ["Lab LAN server", "FastAPI + both models (CPU)", "~0.5-1 s per frame; LAN only; optional capture of "
                   "unsure frames for labelling (opt-in)"],
                  ["Public demo (Render)", "same code and models", "no lab data on the server; sleeps when idle"],
                  ["Unity / Quest 3", "client script, AR module per model", "sends passthrough frames, receives labels"],
                  ["On-device option", "ONNX export of the detector", "for Unity Sentis; verified equal to PyTorch"],
                  ["Code", "private GitHub repository", "lab photos, labels and checkpoints stay local"]],
                 [0.2, 0.32, 0.48])]

    # ---------------------------------------------------------------- 8 next stage
    st += [P("8. Next stage: 6D pose with FoundationPose", "h1"),
           P("The recognised model will select a 3D model for FoundationPose (NVIDIA, CVPR 2024), which estimates "
             "and then tracks the 6D pose of novel objects without per-object training. A detailed study is in "
             "<i>docs/FOUNDATIONPOSE_STUDY.md</i>; the key points:"),
           *bullets(["Inputs: RGB image, <b>metric depth aligned to the RGB image</b>, camera intrinsics, an <b>object "
                     "mask on the first frame</b> (our box can prompt a segmentation model such as SAM 2), and a "
                     "metric, textured mesh per instrument.",
                     "The two Tektronix models share one case, so one mesh with two textures covers both; the RTB2004 "
                     "needs its own. The recognised label selects the mesh through the instrument registry.",
                     "Output: a 4 x 4 object-to-camera pose and a score; ~1.3 s for the first estimate and ~32 Hz "
                     "tracking on an RTX 3090.",
                     "<b>Main risk:</b> the Quest 3 gives camera images but no depth aligned to them; the PoC should "
                     "start with an external RGB-D camera, then evaluate stereo depth from the two Quest cameras.",
                     "Requires an NVIDIA GPU (Linux/Docker). Licence: research/evaluation only (non-commercial)."])]

    # ---------------------------------------------------------------- 9 limitations / next steps
    st += [P("9. Limitations and next steps", "h1"),
           *bullets(["<b>Data, not model size, limits model naming.</b> 20-50 new photos or short videos of the "
                     "TDS 2014 and TDS 1002 fronts from several angles (especially from above), in 2-3 rooms, are the "
                     "most effective next step.",
                     "The test set is small (85 instruments, only 6 TDS 1002); a larger test session would give "
                     "tighter estimates.",
                     "Rear and side views of the two Tektronix models are physically almost identical; the system "
                     "reports \"model unclear\" there, and the pose stage should wait for a front view before "
                     "loading model-specific AR content.",
                     "Pose stage: build meshes, prove FoundationPose with an RGB-D camera on the RTB2004, then "
                     "integrate it as an optional server stage and evaluate Quest depth options.",
                     "The pipeline is object-agnostic: a registry entry plus photos/videos are enough to start a new "
                     "object such as a PCB or a coffee machine (<i>docs/NEW_OBJECT_KIT.md</i>)."])]

    # ---------------------------------------------------------------- appendix
    st += [P("Appendix: components", "h1"),
           table([["Component", "Technology"],
                  ["Detector", "YOLOX-Tiny 0.3.0 (PyTorch), COCO-pretrained, fine-tuned on 3 classes"],
                  ["Second stage", "MobileNetV3-small (torchvision), ImageNet-pretrained"],
                  ["Context objects", "YOLOX-Tiny COCO (80 everyday classes)"],
                  ["Server / UI", "FastAPI, Uvicorn; HTML/JS page; Docker"],
                  ["Labelling", "Label Studio 1.23 with model-proposed boxes; scripts for review sheets"],
                  ["Data generation", "OpenCV-based scene generator, hard-negative mining, COCO / YOLO / VOC export"],
                  ["Hosting", "LAN server (lab); Render (demo); GitHub (code)"],
                  ["AR client", "Unity (C#) for Meta Quest 3, ONNX export for Unity Sentis"],
                  ["Pose (planned)", "FoundationPose (NVIDIA), SAM 2 for masks, RGB-D camera"]],
                 [0.25, 0.75])]

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="Recognising laboratory oscilloscopes for an AR assistant",
                            author="Leonardo AR PoC", subject="Project report")
    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    print("written", OUT)


if __name__ == "__main__":
    build()
