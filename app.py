# app.py
import streamlit as st
from transformers import pipeline
import spacy
import re
import docx
from PyPDF2 import PdfReader
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# -------------------------
# Load models (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    # summarizer: will auto-download default if needed
    summarizer = pipeline("summarization")
    nlp = spacy.load("en_core_web_sm")
    return summarizer, nlp

summarizer, nlp = load_models()

# -------------------------
# Helper functions
# -------------------------
def extract_section(text: str, start_keyword: str, end_keywords: list):
    """
    Extract block of text after a heading (start_keyword) until the next known heading or end.
    Keeps bullets and newlines.
    """
    # Build regex to find heading and capture until next heading or end
    end_group = "|".join([re.escape(k) for k in end_keywords])
    pattern = rf"(?ims){re.escape(start_keyword)}\s*[:\-]?\s*(.*?)(?=(?:{end_group})\s*[:\-]|\Z)"
    m = re.search(pattern, text)
    if m:
        return m.group(1).strip()
    return ""

def remove_patient_info_from_block(block: str):
    """
    Remove common patient info lines to avoid them ending up in diagnosis/medication/advice.
    """
    lines = block.splitlines()
    filtered = []
    for line in lines:
        l = line.strip()
        # skip lines that look like patient name, age, gender, date, phone or vitals that belong in different sections
        if re.search(r"^(patient\s*name|name)\b", l, re.I): 
            continue
        if re.search(r"\b(age|dob|date of birth)\b", l, re.I):
            continue
        if re.search(r"\b(gender|sex)\b", l, re.I):
            continue
        if re.search(r"^(date)\b", l, re.I):
            continue
        if re.search(r"\b(bp|blood pressure|pulse|bpm|bmi)\b", l, re.I) and len(l.split()) < 6:
            # small vitals line — often not needed in medication block (we keep for diagnosis maybe)
            continue
        filtered.append(l)
    return "\n".join(filtered).strip()

def dedupe_preserve_order(text: str):
    """
    Deduplicate lines or sentences while preserving order.
    """
    seen = set()
    out_lines = []
    # split on lines and on bullet separators
    lines = [ln.strip() for ln in re.split(r"\n|\r|\u2022|- ", text) if ln.strip()]
    for ln in lines:
        key = re.sub(r"\s+", " ", ln).lower()
        if key not in seen:
            seen.add(key)
            out_lines.append(ln)
    return "\n".join(out_lines)

def safe_summarize(text, max_len=120, min_len=30):
    """
    Call summarizer with safe try/except and return text.
    """
    try:
        out = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return out[0]["summary_text"].strip()
    except Exception as e:
        # fallback: return first 2 sentences
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        return " ".join(sentences[:2]).strip()

def generate_pdf_bytes(title, extractive, abstractive, diagnosis, medication, advice):
    """
    Create a PDF in-memory and return bytes (ReportLab).
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 0.7 * inch
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, y, title)
    y -= 0.35 * inch

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Extractive Summary:")
    y -= 0.22 * inch
    c.setFont("Helvetica", 11)
    text = c.beginText(margin, y)
    for line in (extractive or "Not found").splitlines():
        text.textLine(line)
        y -= 0.18 * inch
        if y < margin:
            c.drawText(text); c.showPage(); text = c.beginText(margin, height - margin); y = height - margin
    c.drawText(text)

    y = y - 0.12 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Abstractive Summary:")
    y -= 0.22 * inch
    c.setFont("Helvetica", 11)
    text = c.beginText(margin, y)
    for line in (abstractive or "Not found").splitlines():
        text.textLine(line)
        y -= 0.18 * inch
        if y < margin:
            c.drawText(text); c.showPage(); text = c.beginText(margin, height - margin); y = height - margin
    c.drawText(text)

    # Structured info
    y -= 0.25 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Diagnosis:")
    y -= 0.22 * inch
    c.setFont("Helvetica", 11)
    text = c.beginText(margin, y)
    for line in (diagnosis or "Not found").splitlines():
        text.textLine(line)
        y -= 0.18 * inch
        if y < margin:
            c.drawText(text); c.showPage(); text = c.beginText(margin, height - margin); y = height - margin
    c.drawText(text)

    y -= 0.25 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Medication:")
    y -= 0.22 * inch
    c.setFont("Helvetica", 11)
    text = c.beginText(margin, y)
    for line in (medication or "Not found").splitlines():
        text.textLine(line)
        y -= 0.18 * inch
        if y < margin:
            c.drawText(text); c.showPage(); text = c.beginText(margin, height - margin); y = height - margin
    c.drawText(text)

    y -= 0.25 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Advice:")
    y -= 0.22 * inch
    c.setFont("Helvetica", 11)
    text = c.beginText(margin, y)
    for line in (advice or "Not found").splitlines():
        text.textLine(line)
        y -= 0.18 * inch
        if y < margin:
            c.drawText(text); c.showPage(); text = c.beginText(margin, height - margin); y = height - margin
    c.drawText(text)

    # Footer dev credit
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(width / 2, margin / 2, "Developed by Wajeeha Sahar — wsahar527@gmail.com")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

# -------------------------
# UI Styling (dark-blue / white)
# -------------------------
st.set_page_config(page_title="AI POWERED MEDICAL REPORT SUMMARIZER", layout="wide")

# --- Light Blue Theme CSS ---
st.markdown("""
    <style>
        /* Background & Text */
        body, .stApp {
            background-color: #E3F2FD;  /* Light blue background */
            color: #0D47A1;  /* Deep blue text */
        }

        /* Header */
        h1, h2, h3, h4 {
            color: #0D47A1 !important;
            text-align: center;
        }

        /* Radio Buttons and Labels */
        .stRadio > label, .stMarkdown {
            color: #0D47A1 !important;
        }

        /* Text Area */
        textarea {
            background-color: #FFFFFF !important;
            border: 2px solid #90CAF9 !important;
            color: #0D47A1 !important;
            border-radius: 10px !important;
        }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(90deg, #64B5F6, #2196F3);
            color: white;
            font-weight: 600;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
            transition: 0.3s ease;
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #42A5F5, #1976D2);
            box-shadow: 0 0 10px rgba(33,150,243,0.4);
        }

        /* Card Styling */
        .card {
            border-radius: 15px;
            padding: 15px;
            color: white;
            margin-bottom: 15px;
            font-size: 15px;
        }

        /* Footer */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: linear-gradient(90deg, #E3F2FD, #64B5F6);
            color: #0D47A1;
            text-align: center;
            padding: 10px 0;
            font-size: 15px;
            font-weight: 500;
            box-shadow: 0 -2px 8px rgba(33, 150, 243, 0.3);
            z-index: 999;
        }

        .footer a {
            color: #1565C0;
            text-decoration: none;
            font-weight: 600;
        }

        .footer a:hover {
            color: #0D47A1;
            text-shadow: 0 0 10px #64B5F6;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("<div class='big-title'>💙 AI POWERED MEDICAL REPORT SUMMARIZER 💙</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Paste text or upload a .docx / .pdf → click Summarize to get extractive & abstractive summaries plus diagnosis, medication, advice.</div>", unsafe_allow_html=True)

# -------------------------
# Inputs
# -------------------------
input_type = st.radio("Select Input Type:", ["Text", "Document (.docx)", "PDF"], horizontal=True, key="input_type_radio")

text_input = ""
if input_type == "Text":
    text_input = st.text_area("📝 Paste or write your medical report here:", height=250, key="text_input_area")
elif input_type == "Document (.docx)":
    uploaded_file = st.file_uploader("Upload a DOCX file", type=["docx"], key="docx_uploader")
    if uploaded_file:
        doc = docx.Document(uploaded_file)
        text_input = "\n".join([p.text for p in doc.paragraphs])
        st.success("Loaded DOCX text (you can edit above).")
elif input_type == "PDF":
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")
    if uploaded_pdf:
        try:
            reader = PdfReader(uploaded_pdf)
            pages_text = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
            text_input = "\n".join(pages_text)
            st.success("Loaded PDF text (you can edit above).")
        except Exception as e:
            st.error("Could not extract PDF text. Please try a different file.")

# Keep last results in session_state so Download button can access them
if "extractive_summary" not in st.session_state:
    st.session_state["extractive_summary"] = ""
if "abstractive_summary" not in st.session_state:
    st.session_state["abstractive_summary"] = ""
if "diagnosis_text" not in st.session_state:
    st.session_state["diagnosis_text"] = ""
if "medication_text" not in st.session_state:
    st.session_state["medication_text"] = ""
if "advice_text" not in st.session_state:
    st.session_state["advice_text"] = ""

# -------------------------
# Summarize action
# -------------------------
col_btn = st.container()
with col_btn:
    if st.button("🔍 Summarize Report", key="summarize_button"):
        if not text_input or not text_input.strip():
            st.warning("Please paste text or upload a document/pdf first.")
        else:
            # show progress bar while working
            prog = st.progress(0, text="Initializing...")
            try:
                prog.progress(10, text="Preparing text...")
                # Clean up repeated whitespace
                raw_text = re.sub(r"\r\n", "\n", text_input).strip()

                prog.progress(25, text="Running extractive summarization...")
                extractive = safe_summarize(raw_text, max_len=120, min_len=30)

                prog.progress(55, text="Running abstractive summarization...")
                abstractive = safe_summarize(raw_text, max_len=160, min_len=40)

                prog.progress(70, text="Extracting structured sections...")
                # Try heading-based extraction
                diagnosis_block = extract_section(raw_text, "Diagnosis", ["Medical History", "Medication", "Advice", "Plan", "Recommendations", "Examination"])
                medication_block = extract_section(raw_text, "Medication", ["Advice", "Plan", "Recommendations", "Prescription", "Medications", "Allergies"])
                advice_block = extract_section(raw_text, "Advice", ["Plan", "Recommendations", "Follow", "Conclusion", "Signature"])

                # If any section empty, try alternative headings / NER heuristics
                if not diagnosis_block:
                    alt = extract_section(raw_text, "Assessment", ["Plan", "Medication", "Advice"])
                    if alt:
                        diagnosis_block = alt

                # Remove patient-info lines from blocks
                diagnosis_block = remove_patient_info_from_block(diagnosis_block)
                medication_block = remove_patient_info_from_block(medication_block)
                advice_block = remove_patient_info_from_block(advice_block)

                # If medication not found by heading, look for Rx-style lines (med names + doses)
                if not medication_block:
                    meds = re.findall(r"\b([A-Z][a-zA-Z0-9\-\+]{2,}\s*\d{1,4}\s*(?:mg|mcg|g|units|IU|tablet|tab|capsule|ml|once|twice|daily)?)", raw_text, re.I)
                    medication_block = "\n".join(meds[:8]) if meds else ""

                # Clean duplicates and trim
                diagnosis_block = dedupe_preserve_order(diagnosis_block) if diagnosis_block else ""
                medication_block = dedupe_preserve_order(medication_block) if medication_block else ""
                advice_block = dedupe_preserve_order(advice_block) if advice_block else ""

                prog.progress(95, text="Finalizing...")
                # Save to session state
                st.session_state["extractive_summary"] = extractive
                st.session_state["abstractive_summary"] = abstractive
                st.session_state["diagnosis_text"] = diagnosis_block
                st.session_state["medication_text"] = medication_block
                st.session_state["advice_text"] = advice_block

                prog.progress(100, text="Done.")
                st.success("✅ Summaries and details generated successfully!")
            except Exception as e:
                st.error(f"Error while summarizing: {e}")
                prog.empty()

# -------------------------
# Show results (if present)
# -------------------------
if st.session_state["extractive_summary"] or st.session_state["abstractive_summary"]:
    # two-column layout for summaries
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("🧠 Extractive Summary")
        st.markdown(f"<div class='reportbox'>{st.session_state['extractive_summary']}</div>", unsafe_allow_html=True)
    with col2:
        st.subheader("✨ Abstractive Summary")
        st.markdown(f"<div class='reportbox'>{st.session_state['abstractive_summary']}</div>", unsafe_allow_html=True)

    st.markdown("### 📋 Structured Information")

    # Diagnosis card (red)
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg,#ff4d4f,#c40000);
                    color:white;border-radius:12px;padding:14px;margin-bottom:10px;box-shadow: 0 6px 20px rgba(196,0,0,0.3);">
            <h4 style="margin:0">🩺 Diagnosis</h4>
            <pre style="white-space:pre-wrap;font-size:14px;">{st.session_state['diagnosis_text'] or 'Not found'}</pre>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Medication card (green)
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg,#00c853,#007e33);
                    color:white;border-radius:12px;padding:14px;margin-bottom:10px;box-shadow: 0 6px 20px rgba(0,150,136,0.25);">
            <h4 style="margin:0">💊 Medication</h4>
            <pre style="white-space:pre-wrap;font-size:14px;">{st.session_state['medication_text'] or 'Not found'}</pre>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Advice card (purple)
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg,#8e2de2,#4a00e0);
                    color:white;border-radius:12px;padding:14px;margin-bottom:20px;box-shadow: 0 6px 20px rgba(138,43,226,0.25);">
            <h4 style="margin:0">💡 Advice</h4>
            <pre style="white-space:pre-wrap;font-size:14px;">{st.session_state['advice_text'] or 'Not found'}</pre>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Download PDF button (unique key)
    pdf_bytes = None
    try:
        pdf_bytes = generate_pdf_bytes(
            "AI Medical Report Summary",
            st.session_state["extractive_summary"],
            st.session_state["abstractive_summary"],
            st.session_state["diagnosis_text"],
            st.session_state["medication_text"],
            st.session_state["advice_text"]
        )
    except Exception as e:
        st.error("Could not generate PDF: " + str(e))

    if pdf_bytes:
        st.download_button(
            label="⬇️ Download Summary as PDF",
            data=pdf_bytes,
            file_name="medical_summary.pdf",
            mime="application/pdf",
            key="download_pdf_button"
        )

# -------------------------
# Developer footer
# -------------------------
st.markdown(
    """
    <style>
    .dev-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(90deg,#001f4d,#0044cc);
        color: #fff;
        text-align: center;
        padding: 10px 8px;
        font-weight: 600;
        z-index: 9999;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.4);
    }
    .dev-footer a { color: #bdf2ff; text-decoration:none; font-weight:700; }
    </style>
    <div class="dev-footer">
        👩‍💻 Developed by <b>Wajeeha Sahar</b> — 
        <a href="mailto:wsahar527@gmail.com">wsahar527@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True
)

