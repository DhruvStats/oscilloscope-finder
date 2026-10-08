"""Build docs/Oscilloscope_Finder_Guide.pdf - how to start the project and how it works.

    .venv/Scripts/python docs/build_guide.py
"""
import os

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Oscilloscope_Finder_Guide.pdf")
IMG = os.path.join(HERE, "img")

INK = colors.HexColor("#1d1d1b")
MUTED = colors.HexColor("#5f5e5a")
LINE = colors.HexColor("#d3d1c7")
SOFT = colors.HexColor("#f1efe8")
ACCENT = colors.HexColor("#0f6e56")
GREEN = colors.HexColor("#18c27f")
ORANGE = colors.HexColor("#ff8a1f")
MAGENTA = colors.HexColor("#d14bd1")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=26, leading=31,
                            textColor=INK, alignment=0, spaceAfter=6),
    "sub": ParagraphStyle("s", fontName="Helvetica", fontSize=13, leading=18, textColor=MUTED, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, leading=22, textColor=ACCENT,
                         spaceBefore=4, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=INK,
                         spaceBefore=10, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="Helvetica", fontSize=10, leading=14.5, textColor=INK, spaceAfter=6),
    "small": ParagraphStyle("sm", fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=MUTED),
    "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8.5, leading=11, textColor=MUTED,
                          alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
    "b": ParagraphStyle("b", fontName="Helvetica", fontSize=10, leading=14, textColor=INK, leftIndent=12,
                        bulletIndent=2, spaceAfter=2),
    "code": ParagraphStyle("c", fontName="Courier", fontSize=8.6, leading=11.5, textColor=INK,
                           backColor=SOFT, borderPadding=(5, 6, 5, 6), leftIndent=4, rightIndent=4,
                           spaceBefore=3, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9, leading=12, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=INK),
}


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def P(t, s="p"):
    return Paragraph(t, S[s])


def B(items):
    return [Paragraph(t, S["b"], bulletText="\u2022") for t in items]


def code(*lines):
    return Paragraph("<br/>".join(esc(l).replace(" ", "&nbsp;") for l in lines), S["code"])


def table(rows, widths, header=True):
    data = [[Paragraph(c if isinstance(c, str) else str(c), S["cellb" if header and i == 0 else "cell"])
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    t.setStyle(TableStyle(st))
    return t


def img(name, width, caption=None):
    path = os.path.join(IMG, name)
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    out = [Image(path, width=width, height=width * ih / iw)]
    if caption:
        out.append(P(caption, "cap"))
    return out


def pipeline():
    """Flow diagram: photo -> server -> detector (whole + tiles) -> merge + cut-off -> answer -> page/AR."""
    W, H = 170 * mm, 46 * mm
    d = Drawing(W, H)
    boxes = [("Photo / camera", "frame"), ("FastAPI server", "/v1/recognitions"),
             ("YOLOX-Tiny", "whole photo + tiles"), ("Merge + 40% cut-off", "one box per object"),
             ("Answer", "name, confidence, box")]
    bw, gap = 29 * mm, 6.2 * mm
    y = 15 * mm
    for i, (t, s) in enumerate(boxes):
        x = i * (bw + gap)
        d.add(Rect(x, y, bw, 17 * mm, rx=3, ry=3, fillColor=SOFT, strokeColor=ACCENT, strokeWidth=0.8))
        d.add(String(x + bw / 2, y + 10 * mm, t, fontName="Helvetica-Bold", fontSize=7.6, fillColor=INK,
                     textAnchor="middle"))
        d.add(String(x + bw / 2, y + 5 * mm, s, fontName="Helvetica", fontSize=6.6, fillColor=MUTED,
                     textAnchor="middle"))
        if i < len(boxes) - 1:
            x0, x1, ym = x + bw + 0.8 * mm, x + bw + gap - 1.2 * mm, y + 8.5 * mm
            d.add(Line(x0, ym, x1, ym, strokeColor=MUTED, strokeWidth=0.9))
            d.add(Polygon([x1, ym, x1 - 1.8 * mm, ym + 1.1 * mm, x1 - 1.8 * mm, ym - 1.1 * mm],
                          fillColor=MUTED, strokeColor=MUTED))
    d.add(String(W / 2, 6 * mm, "Used by: the web page (upload a photo)  and  the Quest/Unity client (live frames)",
                 fontName="Helvetica", fontSize=7.4, fillColor=MUTED, textAnchor="middle"))
    for x, c, n in [(0, GREEN, "R&S RTB2004"), (52 * mm, ORANGE, "Tektronix TDS 2014"),
                    (110 * mm, MAGENTA, "Tektronix TDS 1002")]:
        d.add(Rect(x + 8 * mm, 37 * mm, 4 * mm, 4 * mm, fillColor=c, strokeColor=c))
        d.add(String(x + 14 * mm, 38 * mm, n, fontName="Helvetica", fontSize=7.8, fillColor=INK))
    return d


def flow():
    """Simple story: photos -> labels -> dataset -> training -> test -> server -> users."""
    W, H = 170 * mm, 40 * mm
    d = Drawing(W, H)
    steps = [("1 Photos", "your lab photos"), ("2 Labels", "boxes, checked"), ("3 Dataset", "renders + cut-outs"),
             ("4 Training", "YOLOX-Tiny, CPU"), ("5 Test", "14 real photos"), ("6 Server", "local or Render"),
             ("7 Users", "web page, Quest")]
    bw, gap = 21.5 * mm, 3.25 * mm
    y = 12 * mm
    for i, (t, sub) in enumerate(steps):
        x = i * (bw + gap)
        d.add(Rect(x, y, bw, 16 * mm, rx=3, ry=3, fillColor=SOFT, strokeColor=ACCENT, strokeWidth=0.8))
        d.add(String(x + bw / 2, y + 9.5 * mm, t, fontName="Helvetica-Bold", fontSize=7.4, fillColor=INK,
                     textAnchor="middle"))
        d.add(String(x + bw / 2, y + 4.5 * mm, sub, fontName="Helvetica", fontSize=5.9, fillColor=MUTED,
                     textAnchor="middle"))
        if i < len(steps) - 1:
            x0, x1, ym = x + bw + 0.3 * mm, x + bw + gap - 0.3 * mm, y + 8 * mm
            d.add(Polygon([x1, ym, x0, ym + 1.3 * mm, x0, ym - 1.3 * mm], fillColor=MUTED, strokeColor=MUTED))
    d.add(String(W / 2, 4 * mm, "If the test score is better than the current model, the new model is deployed; "
                 "otherwise the old one stays.", fontName="Helvetica", fontSize=7.2, fillColor=MUTED,
                 textAnchor="middle"))
    return d


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "Oscilloscope Finder - Leonardo AR PoC")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=18 * mm, title="Oscilloscope Finder - Guide",
                            author="Oscilloscope Finder project", subject="How to start and how it works")
    fw = A4[0] - 36 * mm
    st = []

    # ---------------------------------------------------------------- cover / overview
    st += [P("Oscilloscope Finder", "title"),
           P("How to start the project, and how it works", "sub"),
           P("<b>What it does.</b> You give it a photo (or a live camera frame). It finds the three oscilloscopes "
             "of the Leonardo AR proof-of-concept, names the exact model and draws a box around each one: "
             "<b>Rohde &amp; Schwarz RTB2004</b>, <b>Tektronix TDS 2014</b> and <b>Tektronix TDS 1002</b>. "
             "Everything runs on a normal CPU, on the local network, without cloud AI services."),
           Spacer(1, 4), pipeline(), Spacer(1, 6)]
    st += img("result_office.jpg", fw, "A real office photo processed by the app: both oscilloscopes are found and "
                                       "named, everything else is dimmed.")
    st.append(table([
        ["Part", "What it is"],
        ["Detector", "YOLOX-Tiny (the model named in the Leonardo notes), fine-tuned for the 3 oscilloscopes"],
        ["Server", "FastAPI with the Leonardo contract: GET /health, POST /v1/recognitions"],
        ["Web page", "upload / drag / paste a photo, see and download the labelled image"],
        ["Labelling", "Label Studio project \"PoC oscilloscopes\" with pre-drawn boxes"],
        ["Headset", "Unity / Quest client script using the same API"],
        ["Hosting", "local PC (main use); optional public demo on Render via Docker"],
    ], [32 * mm, fw - 32 * mm]))
    st.append(PageBreak())

    # ---------------------------------------------------------------- how to start
    st += [P("1. How to start the project", "h1"),
           P("Main folder: <b>C:\\Users\\WAGH\\oscilloscope-detection</b>. Open PowerShell there for every command "
             "below. Everything is already installed on the lab PC; the first-time setup is only needed on a new "
             "computer.")]
    st += [P("1.1 Start the oscilloscope finder (web app + API)", "h2"),
           code("cd C:\\Users\\WAGH\\oscilloscope-detection",
                ".venv\\Scripts\\python -m uvicorn server.app:app --host 127.0.0.1 --port 8011"),
           P("Open <b>http://localhost:8011</b> in the browser. The first start takes about 20 seconds (the models "
             "load). Stop it with <b>Ctrl+C</b>. Port 8011 is used because 8001 belongs to another project on this PC."),
           P("1.2 Start Label Studio (only when labelling)", "h2"),
           code(".\\labelstudio\\start.ps1"),
           P("Open <b>http://localhost:8080</b>, sign in, and open the project <b>PoC oscilloscopes</b> "
             "(http://localhost:8080/projects/2/data). Stop it with Ctrl+C."),
           P("1.3 Use it from the Quest headset or another PC on the LAN", "h2"),
           code(".venv\\Scripts\\python -m uvicorn server.app:app --host 0.0.0.0 --port 8011"),
           P("Allow port 8011 in the Windows firewall for the private network, then use the PC's LAN address, e.g. "
             "http://192.168.1.20:8011. In Unity, put this address in the <b>Server Url</b> of "
             "<i>unity/OscilloscopeRecognitionClient.cs</i>. Never expose the server to the Internet."),
           P("1.4 Hosted demo (Render)", "h2"),
           P("The GitHub repository <b>DhruvStats/oscilloscope-finder</b> contains a Dockerfile and render.yaml. "
             "Render rebuilds automatically after every push. The free plan sleeps when idle, so the first visit "
             "after a pause takes about a minute. The demo has no password; to add one, set DEMO_PASSWORD in "
             "Render > Environment. The hosted demo sends photos to Render's servers, so real lab use must stay on "
             "the local server."),
           P("1.5 First-time setup on a new computer", "h2"),
           code("git clone https://github.com/DhruvStats/oscilloscope-finder.git oscilloscope-detection",
                "cd oscilloscope-detection",
                "python -m venv .venv",
                ".venv\\Scripts\\pip install torch torchvision ^",
                "    --index-url https://download.pytorch.org/whl/cpu",
                ".venv\\Scripts\\pip install -r requirements-deploy.txt",
                "git clone --depth 1 --branch 0.3.0 ^",
                "    https://github.com/Megvii-BaseDetection/YOLOX.git models\\yolox",
                "python -m venv .labelstudio-venv",
                ".labelstudio-venv\\Scripts\\pip install label-studio"),
           P("Lab photos, datasets and training checkpoints are not on GitHub (lab data); copy them from the lab PC "
             "if you want to retrain. The trained model itself (models/deploy) is in the repository.", "small"),
           P("1.6 Using the web page", "h2")]
    st += B(["<b>Choose image</b>, drag a photo onto the page, paste one with Ctrl+V, or click a sample.",
             "Each oscilloscope gets a coloured box: <font color='#0f9d6b'>green</font> RTB2004, "
             "<font color='#d9711a'>orange</font> TDS 2014, <font color='#b03fb0'>magenta</font> TDS 1002, with the "
             "confidence in %.",
             "<b>Focus on oscilloscopes</b> dims the rest of the photo; <b>Label other objects</b> shows everyday "
             "objects (bottle, chair, tv) in grey when that model is on.",
             "<b>Download labelled image</b> saves the result as JPEG. <b>New image</b> clears the page.",
             "No sliders: the server applies one tested confidence cut-off (40%, see section 3.7)."])
    st.append(PageBreak())

    # ---------------------------------------------------------------- components in simple words
    st += [P("2. What each part does, in simple words", "h1"),
           P("The project is like a small factory. Photos go in at one end; at the other end a program can look "
             "at any new photo and say <i>&quot;that is a Tektronix TDS 2014&quot;</i>. Here is every part, what it "
             "does, and where it lives."),
           flow(), Spacer(1, 4),
           P("The parts that do the recognising", "h2"),
           table([
               ["Part", "What it does, in simple words", "Where"],
               ["<b>Detector model</b> (YOLOX-Tiny)",
                "The &quot;brain&quot;. It looks at a picture and says: here is an oscilloscope, this is its model, "
                "and I am 87% sure. Small enough to run on a normal computer. We taught it our three oscilloscopes.",
                "models/deploy/"],
               ["<b>Everyday-objects model</b> (optional)",
                "A second brain that already knows 80 common things (bottle, chair, tv) and labels them in grey. "
                "On locally, off on Render to save memory.",
                "models/"],
               ["<b>Server</b> (FastAPI)",
                "The &quot;reception desk&quot;. It receives a photo, asks the detector, also looks at zoomed-in "
                "pieces of the photo so small far-away instruments are not missed, removes doubles, throws away "
                "unsure answers (below 40%), and sends back the result.",
                "server/app.py"],
               ["<b>Web page</b>",
                "The &quot;window&quot; for people. Upload or paste a photo, see coloured boxes with names, "
                "download the result.",
                "web/index.html"],
               ["<b>Headset client</b> (Unity / Quest)",
                "Like the web page, but for the AR headset: it sends what the camera sees about once per second "
                "and switches on the right AR content.",
                "unity/"],
           ], [38 * mm, fw - 70 * mm, 32 * mm]),
           P("The parts that teach the brain", "h2"),
           table([
               ["Part", "What it does, in simple words", "Where"],
               ["<b>Photos</b>", "64 photos of the real instruments taken in the lab: the raw material.", "raw/"],
               ["<b>Labels</b>",
                "For every photo: where each instrument is (a box) and which model it is - like the answers to a "
                "school exercise.",
                "raw/real_labels.json"],
               ["<b>Label Studio</b>",
                "A drawing tool in the browser. It shows each photo with the boxes already drawn; a person checks "
                "them and fixes mistakes.",
                "labelstudio/"],
               ["<b>Dataset generator</b>",
                "Makes many practice pictures from few photos: 3D models of the instruments, real instruments cut "
                "out and pasted into new scenes, zoomed copies, plus look-alikes (like a wall socket) that must "
                "<i>not</i> be called an oscilloscope. It places everything itself, so every box is exact.",
                "synth/"],
               ["<b>Training</b>",
                "The &quot;lessons&quot;. The brain studies the practice pictures again and again (20 rounds, about "
                "2 hours) and slowly gets better. 80% of the pictures are for learning, 20% for checking during "
                "training.",
                "training/"],
               ["<b>Test</b>",
                "The &quot;final exam&quot;: 14 real photos the brain has never seen. The score tells honestly how "
                "good it is (now 14 of 18 instruments right). A new brain is only used if it scores better.",
                "tools/eval_app.py"],
           ], [38 * mm, fw - 70 * mm, 32 * mm]),
           KeepTogether([P("The parts that put it online", "h2"),
           table([
               ["Part", "What it does, in simple words", "Where"],
               ["<b>GitHub</b>",
                "The online safe for the code and the trained brain (private). Photos stay on the lab PC.",
                "repo oscilloscope-finder"],
               ["<b>Docker</b>",
                "Packs the server, the brain and everything they need into one box that runs the same anywhere.",
                "Dockerfile"],
               ["<b>Render</b>",
                "A hosting service that takes the box from GitHub and puts the web page on the Internet. It "
                "updates by itself after every change. Free plan: sleeps when nobody uses it.",
                "render.yaml"],
               ["<b>Guide and README</b>", "The instructions: this PDF and the README in the main folder.",
                "docs/, README.md"],
           ], [38 * mm, fw - 70 * mm, 32 * mm])]),
           P("Words used in this guide", "h2"),
           table([
               ["Word", "Meaning"],
               ["Box (bounding box)", "the rectangle drawn around an instrument"],
               ["Confidence", "how sure the brain is, from 0% to 100%"],
               ["Threshold / cut-off", "answers below this confidence (40%) are ignored, because they are mostly "
                                      "wrong"],
               ["Training / epoch", "teaching the brain; one epoch = it has seen all practice pictures once"],
               ["Validation / test set", "pictures kept apart to check the brain; the test set is real photos only"],
               ["False alarm", "the brain calls something an oscilloscope that is not one (e.g. a wall socket)"],
               ["Tiles", "zoomed-in pieces of a photo, so small far-away instruments become big enough"],
           ], [40 * mm, fw - 40 * mm])]
    st.append(PageBreak())

    # ---------------------------------------------------------------- how it was built
    st += [P("3. Explanation of the project", "h1"),
           P("3.1 The goal", "h2"),
           P("The Leonardo AR proof-of-concept needs a local server that recognises specific physical instruments "
             "in camera frames from an AR headset, so the matching AR module can be shown. Generic detectors only "
             "know classes like \"tv\" or \"microwave\", and the two Tektronix models share the same case, so a "
             "dedicated model had to be trained."),
           P("3.2 The data", "h2")]
    st += B(["<b>64 lab phone photos</b> of the three instruments (front, back, sides, top, bottom, desk scenes).",
             "<b>Batch of 8 October 2026:</b> 39 more photos and 3 walk-around videos, turned into 284 sharp frames "
             "(blurry and near-identical frames dropped). Together: about 350 labelled real images.",
             "<b>1 openly licensed photo</b> of a TDS 1002 (Wikimedia Commons, CC BY-SA 2.5, credited) and a few "
             "R&amp;S press images of the RTB2004, recorded with their source in raw/web/provenance.csv.",
             "Web shop and eBay photos were not used: they are copyrighted and often show newer variants "
             "(TDS2014B/C, TDS1002B) with different front panels."])
    st += [P("3.3 Making many training images from few photos", "h2"),
           P("Real training needs hundreds of labelled images per instrument. Three techniques multiply the real "
             "images, all with exact boxes; the current dataset has <b>1,812 training and 452 validation images "
             "(over 700 boxes per instrument)</b>, saved as COCO, YOLO and Pascal VOC:")]
    st += B(["<b>3D models:</b> the photos of each side were straightened and wrapped onto a box with the real "
             "dimensions, so each instrument can be rendered from any angle (also lying on its side).",
             "<b>Cut and paste:</b> each instrument is cut out of the real photos and pasted into new desk scenes - "
             "real pixels, new positions, sizes, lighting, partly hidden by clutter.",
             "<b>Real crops:</b> every real photo is also used at several zoom levels; photos with the TDS 2014 "
             "(the scarcest model) more often.",
             "Clutter (bottles, backpack, calendar, chair, boards) and look-alikes that caused false alarms (wall "
             "socket, cupboard sign) are pasted without boxes, so the model learns they are not oscilloscopes."])
    st += img("dataset_preview.jpg", fw, "Generated training images: boxes are known exactly because the program "
                                         "placed every instrument itself.")
    st += [P("3.4 Labels", "h2"),
           P("Generated images are labelled automatically. Real photos were pre-labelled by the model, then every "
             "box was checked against the photo; the identity of each Tektronix was confirmed from the front panel "
             "(<b>TDS 2014: coloured buttons, 5 inputs; TDS 1002: grey buttons, 3 inputs</b>) or from the back "
             "(the TDS 1002 has a port module). Wrong or missing boxes were drawn by hand. The result is in "
             "raw/real_labels.json and is loaded into Label Studio for human review (section 4). New batches are "
             "reviewed the same way with tools/review_batch.py (numbered candidate boxes per image), "
             "tools/review_crops.py (close-ups to tell the models apart) and tools/apply_review.py (the decisions).")]
    st.append(PageBreak())

    st += [P("3.5 Training", "h2"),
           P("Detector: <b>YOLOX-Tiny</b> (416 x 416 input), starting from the official COCO weights (release 0.3.0, "
             "commit 41977848, checkpoint SHA-256 9de513de...). The official YOLOX trainer requires an NVIDIA GPU; "
             "this PC has none, so a CPU training loop (training/train_cpu.py) reuses YOLOX's own experiment file, "
             "augmentation, loss and learning-rate schedule; it uses an NVIDIA GPU automatically when one is "
             "present. On this PC one run takes 2-3 hours, on the workstation minutes "
             "(tools/package_for_workstation.py makes a one-zip bundle)."),
           table([["Split", "Content", "Purpose"],
                  ["Train (80%)", "generated scenes + real training photos and their crops", "learning"],
                  ["Validation (20%)", "generated scenes", "choosing the best epoch"],
                  ["Test", "23 real images never used in training: 14 lab photos (all 3 models, front/back/side, "
                           "wide office, far bench) + 9 frames from a lab the model has never seen",
                   "the honest final score"]],
                 [32 * mm, 95 * mm, fw - 127 * mm]),

           P("3.6 Finding small, distant instruments", "h2"),
           P("The model looks at a 416 x 416 version of the photo, so in a wide room shot an oscilloscope shrinks "
             "to about 50 pixels and is missed. The server therefore also looks at overlapping zoomed-in tiles of "
             "the photo and merges the results, keeping one box per object (\"tiled detection\")."),
           P("3.7 Why there is a confidence threshold", "h2"),
           P("For every photo the model proposes many candidate boxes, each with a confidence score. Most are noise "
             "(on one photo a wall socket scored 78%, random spots 5-20%). A cut-off is therefore necessary, but it "
             "does not need to be a slider: the server applies one value, <b>40%</b>, chosen on the real test photos "
             "(results were identical at 30%, 40% and 50%). It can be changed with TARGET_MIN_CONF without code "
             "changes."),
           P("3.8 Results on real photos", "h2"),
           table([["Model version", "Right model + box", "Wrong name", "Missed", "False alarms"],
                  ["v2 - 3D renders + real cut-outs (14 photos)", "10 / 18", "6", "2", "1"],
                  ["v3 - + wide scenes, far / rotated scopes (14 photos)", "14 / 18", "4", "0", "1"],
                  ["v3 on the new 23-image test", "21 / 35", "7", "7", "1"],
                  ["v4 - + 8 Oct batch, 1,812 training images", "21 / 35", "9", "5", "3"],
                  ["<b>v5 (30 epochs) + Tektronix check (deployed)</b>", "<b>26 / 35</b>", "<b>4</b>", "5", "4"],
                  ["<b>Default mode: just \"Oscilloscope\" (v5)</b>", "<b>30 / 35</b>", "-", "5", "4"]],
                 [78 * mm, 28 * mm, 22 * mm, 18 * mm, fw - 146 * mm]),
           P("Scored the way the app works (whole photo + tiles, 40% cut-off) on real images never used in training. "
             "The 23-image test includes 9 frames from a lab the model has never seen, so it is harder than the "
             "first 14-photo test. v4 equals v3 at 40% and beats it at 30%, 50% and 60% (22, 20, 20 vs 21, 18, 18), "
             "and misses fewer instruments; telling TDS 2014 from TDS 1002 is still the main error.", "small"),
           Spacer(1, 4)]
    row = Table([[img("result_bench.jpg", fw * 0.56)[0], img("result_tds1002.jpg", fw * 0.40)[0]]],
                colWidths=[fw * 0.58, fw * 0.42])
    row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    st += [row, P("Left: TDS 2014 and RTB2004 side by side. Right: TDS 1002 on the ARCADIA bench.", "cap")]
    st.append(PageBreak())

    # ---------------------------------------------------------------- labelling + retraining
    st += [P("4. Labelling with Label Studio", "h1"),
           P("Same approach as the joystick dataset in the Leonardo notes: Label Studio runs locally in its own "
             "isolated environment, its database stays in labelstudio/data, and photos are served from raw/ "
             "without copying or uploading. The project <b>PoC oscilloscopes</b> contains the 64 first photos; the 291 "
             "images of the 8 October batch are added with add_tasks.py (below). Every image opens with its "
             "reviewed boxes pre-drawn.")]
    st += B(["Open a photo: the boxes appear as a prediction. Adjust, delete, or draw new ones with the keys "
             "<b>1</b> RTB2004, <b>2</b> TDS 2014, <b>3</b> TDS 1002.",
             "Box every target instrument you can see, even partly. Do not box other instruments, monitors, "
             "sockets or signs.",
             "Set the status (done / no target oscilloscope / unsure which model / exclude) and <b>Submit</b>.",
             "When finished: <b>Export > JSON</b>. The rules are in labelstudio/GUIDELINES.md."])
    st += [P("Add new photos", "h2"),
           code("# new photos: model-proposed boxes;  reviewed batch: its checked boxes",
                ".venv\\Scripts\\python labelstudio\\make_tasks.py --new raw\\new_photos --only-new",
                ".venv\\Scripts\\python labelstudio\\make_tasks.py --only-new --labelled raw\\batch_2026-10-08",
                "$env:LS_TOKEN = \"<token from Account & Settings>\"",
                ".labelstudio-venv\\Scripts\\python labelstudio\\add_tasks.py"),
           P("add_tasks.py adds the tasks to the existing project and skips images that are already there. Keep "
             "the token out of chats and shared files. For a team, start Label Studio with "
             "labelstudio\\start_team.ps1 (LAN only, invite-only sign-up)."),
           P("5. Retraining after new labels", "h1"),
           code(".venv\\Scripts\\python labelstudio\\import_export.py C:\\path\\to\\export.json",
                ".venv\\Scripts\\python synth\\gen3.py --per-class 300 --real-labels raw\\real_labels.json",
                ".venv\\Scripts\\python tools\\export_formats.py          # YOLO + VOC copies",
                ".venv\\Scripts\\python training\\train_cpu.py --exp yolox_tiny_osc3 --epochs 12 ^",
                "    --init models\\deploy\\yolox_tiny_osc3.pth",
                ".venv\\Scripts\\python tools\\eval_app.py models\\training\\yolox_tiny_osc3\\best_ckpt.pth"),
           P("If the score beats the deployed model, copy best_ckpt.pth to models\\deploy\\yolox_tiny_osc3.pth, "
             "commit and push; Render redeploys and the local server picks it up on restart. Keep the PC "
             "plugged in and awake during training (about 2 hours)."),
           P("6. Folder map", "h1"),
           table([["Folder", "Content", "In git"],
                  ["server/, web/", "detector API and web page", "yes"],
                  ["training/, synth/, tools/", "training loop, dataset generator, labelling and test tools", "yes"],
                  ["labelstudio/", "Label Studio setup, interface, guidelines (database excluded)", "yes"],
                  ["config/", "instruments.yaml: the list of recognised instruments", "yes"],
                  ["unity/", "Quest / Unity client", "yes"],
                  ["models/deploy/, models/export/", "deployed weights; ONNX export for edge devices",
                   "yes / no"],
                  ["raw/, datasets/, models/training/", "photos, labels, generated datasets, checkpoints", "no"],
                  ["legacy-yolov8/", "first YOLOv8 + OCR version", "code only"],
                  ["media/, _archive/", "360/3D videos and their scripts; old copy - kept aside", "no"],
                  ["docs/", "this guide and the script that builds it", "yes"]],
                 [45 * mm, fw - 65 * mm, 20 * mm])]

    st += [P("6b. Long-term features (self-hosted, data stays in the lab)", "h2"),
           table([["Need", "What is in the project"],
                  ["Add instruments", "config/instruments.yaml drives server, page, generator, training and the Label "
                                      "Studio interface (tools/registry_sync.py)"],
                  ["Team labelling", "labelstudio/start_team.ps1 - Label Studio on the lab network, invite-only"],
                  ["Fast retraining", "GPU used automatically; tools/package_for_workstation.py for the NVIDIA PC"],
                  ["Learning from real use", "CAPTURE_MODE=on (lab server only): unsure frames from clients that "
                                             "opted in + \"Report wrong result\" -> raw/captures -> Label Studio"],
                  ["Edge / headset", "tools/export_onnx.py -> ONNX for Unity Sentis, ONNX Runtime, OpenVINO, "
                                     "TensorRT (checked against PyTorch)"]],
                 [38 * mm, fw - 38 * mm])]
    st.append(PageBreak())

    # ---------------------------------------------------------------- limits / troubleshooting
    st += [P("7. Known limits and next steps", "h1")]
    st += B(["<b>TDS 2014 vs TDS 1002</b> remains the hardest case: the two share one case and differ mainly in "
             "the front buttons. More real TDS 2014 photos from new places help most.",
             "The test set (23 images) is still small; test with real Quest camera frames before relying on it.",
             "The 3D models are boxes, so unusual angles are approximate; real photos always beat renders.",
             "Licences: lab photos are internal data; the R&amp;S press images and the CC BY-SA photo must pass "
             "the Leonardo licence check before any use beyond the PoC."])
    st += [P("8. Troubleshooting", "h1"),
           table([["Problem", "Fix"],
                  ["Page does not open on 8011", "Start the server (1.1); wait ~20 s for the models to load."],
                  ["\"Port in use\"", "Another program uses the port: pick another, e.g. <font name='Courier'>--port 8012</font>."],
                  ["No oscilloscope found", "Use a closer or sharper photo; the instrument must be one of the "
                                            "three models and reasonably visible."],
                  ["Label Studio shows broken images", "Start it with labelstudio\\start.ps1 (it sets the photo "
                                                       "folder permissions)."],
                  ["setup_project.py: 401", "Copy the token again from Account &amp; Settings and set LS_TOKEN in "
                                            "the same PowerShell window."],
                  ["Render page slow to load", "Free plan was asleep; the first visit takes about a minute."],
                  ["Training stopped", "The PC slept or the session restarted; the best checkpoint so far is "
                                       "saved in models/training/yolox_tiny_osc3/."]],
                 [52 * mm, fw - 52 * mm]),
           Spacer(1, 10),
           P("Repository: github.com/DhruvStats/oscilloscope-finder (private). Main folder: "
             "C:\\Users\\WAGH\\oscilloscope-detection. This guide: docs/Oscilloscope_Finder_Guide.pdf, rebuilt "
             "with  .venv\\Scripts\\python docs\\build_guide.py", "small")]

    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    print("written", OUT)


if __name__ == "__main__":
    build()
