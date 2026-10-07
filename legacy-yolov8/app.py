"""Web frontend: upload any image, find the oscilloscope(s), show brand and model number.

    streamlit run app.py
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from oscilloscope_pipeline import DEFAULT_WEIGHTS, OscilloscopeAnalyzer, annotate, to_text

st.set_page_config(page_title="Oscilloscope Detector", layout="wide")


@st.cache_resource
def load_analyzer(use_ocr: bool):
    return OscilloscopeAnalyzer(use_ocr=use_ocr)


st.title("Oscilloscope Detector")
st.caption("Upload a photo. The model finds each oscilloscope, identifies the brand and reads the model number.")

if not Path(DEFAULT_WEIGHTS).exists():
    st.error("No trained model found at models/best.pt. Run `python scripts/train.py` first.")
    st.stop()

with st.sidebar:
    conf = st.slider("Minimum detection confidence", 0.05, 0.95, 0.25, 0.05)
    use_ocr = st.checkbox("Read model number with OCR", value=True)

uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"])
if uploaded is None:
    st.stop()

image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
if image is None:
    st.error("Could not read that image.")
    st.stop()

with st.spinner("Analysing..." + (" (first OCR run downloads its model, ~1 min)" if use_ocr else "")):
    result = load_analyzer(use_ocr).analyze(image, uploaded.name, conf=conf)

left, right = st.columns([3, 2])
left.image(cv2.cvtColor(annotate(image, result), cv2.COLOR_BGR2RGB), use_container_width=True)

with right:
    if not result.detections:
        st.warning("No oscilloscope found in this image.")
    for i, d in enumerate(result.detections, 1):
        st.subheader(f"Oscilloscope {i}")
        st.metric("Brand", d.brand)
        st.metric("Model number", d.model_number)
        st.write(f"Detection confidence: **{d.confidence:.0%}**")
        if d.model_source == "OCR":
            st.success(f"Model number read from the panel (match score {d.ocr_score:.0f}/100)")
        elif use_ocr:
            st.info("Model number text not readable; using the detector's class instead.")
        if d.ocr_text:
            with st.expander("Raw OCR text"):
                st.code(d.ocr_text)

    st.download_button(
        "Download result .txt",
        to_text(result),
        file_name=f"{Path(uploaded.name).stem}_result.txt",
        mime="text/plain",
    )
