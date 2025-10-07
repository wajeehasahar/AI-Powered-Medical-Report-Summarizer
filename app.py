# app.py
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from docx import Document
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import io
import os
import re
from pdf2image import convert_from_bytes
import numpy as np


# ---------------------------
# Page config & style
# ---------------------------
st.set_page_config(page_title="Medical Report Summarizer", page_icon="🩺", layout="wide")
# Simple light styling for a clean, professional look
st.markdown(
    """
    <style>
    .reportview-container {
        background: #ffffff;
    }
    .stButton>button {
        background-color: #0b66c3;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .stDownloadButton>button {
        background-color: #04a777;
        color: white;
        border-radius: 8px;
    }
    .stSidebar .sidebar-content {
        background: #f8fafd;
    }
    h1 { color: #0b66c3; }
    .small { font-size:0.9rem; color: #444444; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 Medical Report Summarizer")
st.write("Professional, clean summaries from clinical notes — supports typed text, TXT/DOCX, PDF and scanned images (OCR).")

# ---------------------------
# Load models (cached)
# ---------------------------
@st.cache_resource
def load_summarizers():
    # Abstractive (medical-preferred if available)
    try:
        abstractive = pipeline("summarization", model="ccdv/pubmed-summarization")
    except Exception:
        abstractive = pipeline("summarization", model="facebook/bart-large-cnn")
    # Extractive-like model (shorter phrasing)
    try:
        extractive = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    except Exception:
        extractive = abstractive
    return abstractive, extractive

@st.cache_resource
def load_ner():
    try:
        model_name = "samrawal/bert-base-uncased_clinical-ner"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        return ner_pipe
    except Exception:
        return None

abstractive_summarizer, extractive_summarizer = load_summarizers()
ner_pipeline = load_ner()

# ---------------------------
# OCR helper: preprocess image to improve OCR
# ---------------------------
def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    # Convert to grayscale
    img = pil_img.convert("L")
    # Increase contrast
    img = ImageOps.autocontrast(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.4)
    # Denoise / sharpen slightly
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SHARPEN)
    # Resize small images to improve OCR
    w, h = img.size
    if max(w, h) < 1200:
        scale = 1200 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

# ---------------------------
# Text extraction helpers
# ---------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    # Use pdf2image + OCR fallback for scanned pages, but try text extraction via PyMuPDF first
    text_parts = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            txt = page.get_text("text").strip()
            if txt:
                text_parts.append(txt)
        full_text = "\n".join(text_parts).strip()
        if full_text:
            return full_text
    except Exception:
        pass

    # Fallback: render pages to images and OCR
    try:
        images = convert_from_bytes(pdf_bytes, dpi=300)
        ocr_text = []
        for im in images:
            pre = preprocess_image_for_ocr(im)
            ocr_result = pytesseract.image_to_string(pre, lang=None)
            if ocr_result:
                ocr_text.append(ocr_result)
        return "\n".join(ocr_text).strip()
    except Exception:
        return ""

def extract_text_from_image_file(uploaded_file) -> str:
    image = Image.open(uploaded_file).convert("RGB")
    pre = preprocess_image_for_ocr(image)
    text = pytesseract.image_to_string(pre, lang=None)
    return text.strip()

def extract_text_from_docx_file(uploaded_file) -> str:
    doc = Document(uploaded_file)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()

# ---------------------------
# Entity extraction (NER + simple heuristics)
# ---------------------------
def extract_entities(text: str):
    diagnosis = "Not specified"
    meds = "None mentioned"
    advice_lines = []

    # Use NER if available
    if ner_pipeline:
        try:
            ents = ner_pipeline(text)
            problems = [e["word"].strip() for e in ents if e.get("entity_group","").lower() in ["problem","disease","condition"]]
            drugs = [e["word"].strip() for e in ents if e.get("entity_group","").lower() in ["drug","medication","treatment"]]
            if problems:
                diagnosis = ", ".join(dict.fromkeys([p.capitalize() for p in problems]))
            if drugs:
                meds = ", ".join(dict.fromkeys([d for d in drugs]))
        except Exception:
            pass

    # Heuristic regex for meds (fallback)
    if meds == "None mentioned":
        med_matches = re.findall(r"\b([A-Za-z]+(?:\s?\d+mg|\s?\d+ mg)?)\b", text)
        meds_candidates = [m for m in med_matches if len(m) <= 30 and not m.lower() in ("patient","temperature","fever")]
        if meds_candidates:
            meds = ", ".join(dict.fromkeys(meds_candidates[:6]))

    # Advice detection
    advice_keywords = ["rest", "bed rest", "hydration", "fluids", "paracetamol", "acetaminophen", "follow up", "monitor", "antibiotic", "antibiotics"]
    found = []
    for key in advice_keywords:
        if key in text.lower():
            found.append(key)
    if found:
        advice_lines = dict.fromkeys(found)  # dedupe preserving order via dict keys
        advice = ", ".join(advice_lines)
    else:
        advice = "General rest and follow-up advised"

    return {
        "diagnosis": diagnosis,
        "medications": meds,
        "advice": advice
    }

# ---------------------------
# Summarization functions
# ---------------------------
def summarize_with_model(model, text, max_len=120, min_len=30):
    if not text or not text.strip():
        return ""
    text = text.strip()
    # Short-circuit tiny text
    if len(text.split()) < 20:
        return text if len(text) < 240 else text[:240] + "..."
    # Chunk for long text
    if len(text) > 3000:
        chunks = [text[i:i+2500] for i in range(0, len(text), 2500)]
        out = []
        for c in chunks:
            try:
                res = model(c, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
                out.append(res)
            except Exception:
                out.append(c[:min(240, len(c))])
        return " ".join(out)
    else:
        try:
            result = model(text, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
            return result
        except Exception:
            return text[:max_len*2]

# ---------------------------
# Layout: tabs for cleaner UI
# ---------------------------
tabs = st.tabs(["📝 Type / Paste", "📄 Upload File", "🖼️ Upload Image", "⚙️ Options / About"])

with tabs[0]:
    st.subheader("Type or paste clinical notes")
    typed_text = st.text_area("Enter medical report text here:", height=220, placeholder="Paste a clinical note, discharge summary, or consultation note...")
    st.markdown("You can paste multiple paragraphs. For long reports we will chunk and summarize.")

with tabs[1]:
    st.subheader("Upload DOCX / TXT / PDF")
    uploaded_file = st.file_uploader("Choose a file (txt, docx, pdf)", type=["txt", "docx", "pdf"])
    if uploaded_file:
        # handle file reading
        if uploaded_file.name.lower().endswith(".txt"):
            file_text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif uploaded_file.name.lower().endswith(".docx"):
            file_text = extract_text_from_docx_file(uploaded_file)
        else:  # pdf
            pdf_bytes = uploaded_file.read()
            file_text = extract_text_from_pdf_bytes(pdf_bytes)
        st.text_area("Extracted text (preview)", file_text[:4000], height=250)
    else:
        file_text = ""

with tabs[2]:
    st.subheader("Upload scanned image (JPEG/PNG)")
    uploaded_img = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        try:
            img = Image.open(uploaded_img)
            st.image(img, caption="Uploaded image (preview)", use_column_width=True)
            img_text = extract_text_from_image_file(uploaded_img)
            st.text_area("OCR extracted text (preview)", img_text[:4000], height=250)
        except Exception as e:
            st.error("Could not process image: " + str(e))
            img_text = ""
    else:
        img_text = ""

with tabs[3]:
    st.subheader("Options and Info")
    st.markdown("""
    - **Theme:** Light & clean (professional)
    - **OCR:** Uses Tesseract + preprocessing (contrast, denoise) for better recognition of scanned/blurred pages
    - **Models:** Clinical/ PubMed summarizer (if available) with fallback to BART large
    """)
    st.markdown("System note: Make sure the Tesseract OCR engine is installed on the machine where this app runs (instructions in README).")

# ---------------------------
# Combine whichever input the user provided (priority: typed -> uploaded file -> image)
# ---------------------------
input_text = ""
if 'typed_text' in locals() and typed_text and typed_text.strip():
    input_text = typed_text
elif 'file_text' in locals() and file_text and file_text.strip():
    input_text = file_text
elif 'img_text' in locals() and img_text and img_text.strip():
    input_text = img_text

# ---------------------------
# Action buttons (place them outside tabs for easy access)
# ---------------------------
col_left, col_right = st.columns([3,1])
with col_left:
    if st.button("🧠 Generate & Compare Summaries", key="generate"):
        if not input_text.strip():
            st.warning("Please provide input (type or upload).")
        else:
            with st.spinner("Analyzing... this may take a moment for long documents or images"):
                abs_sum = summarize_with_model(abstractive_summarizer, input_text, max_len=110, min_len=25)
                ext_sum = summarize_with_model(extractive_summarizer, input_text, max_len=110, min_len=25)
                entities = extract_entities(input_text)
            # Display summaries
            st.success("✅ Summaries generated")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🔹 Extractive Summary")
                st.write(ext_sum)
            with c2:
                st.subheader("🔸 Abstractive Summary")
                st.write(abs_sum)

            st.markdown("---")
            st.subheader("🧾 Structured Extracted Info")
            st.markdown(f"**🩺 Diagnosis:** {entities['diagnosis']}")
            st.markdown(f"**💊 Medications:** {entities['medications']}")
            st.markdown(f"**🧘 Advice:** {entities['advice']}")

            # Download
            comp_text = (
                f"EXTRACTIVE SUMMARY:\n{ext_sum}\n\n"
                f"ABSTRACTIVE SUMMARY:\n{abs_sum}\n\n"
                f"DIAGNOSIS: {entities['diagnosis']}\n"
                f"MEDICATIONS: {entities['medications']}\n"
                f"ADVICE: {entities['advice']}\n"
            )
            st.download_button("📥 Download Report", comp_text, file_name="medical_summary.txt")

with col_right:
    st.markdown("### Status")
    st.info("Ready to summarize. Choose input above and click 'Generate & Compare Summaries'.")
    st.markdown("### Tips")
    st.markdown("- For scanned images, good lighting and moderate contrast improves OCR results.")
    st.markdown("- If OCR misses content, try uploading a higher-resolution image (300 DPI).")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Developed with ❤️ by Wajeeha sahar wsahar527@gmail.com — Streamlit + Transformers + Clinical NER + OCR")

