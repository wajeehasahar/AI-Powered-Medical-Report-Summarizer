# 🧠 AI-Powered Medical Report Summarizer

A professional **Streamlit-based AI application** that automatically summarizes medical reports using advanced **transformer models** and **clinical NER (Named Entity Recognition)**.  
It supports **text, document, PDF, image, and scanned handwritten reports** — making it ideal for both doctors and patients.  

---

## 🚀 Features

✅ **Multi-format Input Support**
- ✍️ Enter text manually  
- 📄 Upload `.txt`, `.docx`, or `.pdf` files  
- 🖼️ Upload scanned or handwritten **images** (OCR supported)

✅ **AI Summarization**
- **Abstractive** (AI-generated human-like summary)  
- **Extractive** (data-focused factual summary)

✅ **Clinical Entity Detection**
- Detects diseases, medications, and doctor’s advice  
- Auto-highlights medical keywords like “fever”, “antibiotics”, etc.

✅ **Smart OCR**
- Built-in **Tesseract OCR** for images & scanned reports  
- Automatically enhances blurred or handwritten inputs  

✅ **Modern UI**
- Clean, professional Streamlit interface  
- Quick download of comparison report  

---

## 🏗️ Tech Stack

| Component | Description |
|------------|--------------|
| 🧠 Transformers | BART & PubMed models for summarization |
| 💊 Clinical NER | Extracts diagnosis, drugs, and advice |
| 📄 PyMuPDF, python-docx | Document parsing |
| 🖼️ pytesseract | OCR for image & scanned document text extraction |
| ⚡ Streamlit | Interactive web app framework |
| 🔥 Hugging Face Pipelines | High-quality NLP backends |

---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/wajeehasahar/AI-Powered-Medical-Report-Summarizer.git
cd AI-Powered-Medical-Report-Summarizer
