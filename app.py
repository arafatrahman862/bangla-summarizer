# ==============================================================
# 🧠 Bangla Zero-Shot Summarizer (Lightweight CPU Edition)
# ✨ Works on Streamlit Cloud Free Tier
# ==============================================================

import streamlit as st
import torch, time, warnings, concurrent.futures
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from io import BytesIO
from fpdf import FPDF

# ---------- SUPPRESS WARNINGS ----------
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Bangla Summarizer - Zero Shot",
    page_icon="🧠",
    layout="wide",
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top left, #0b132b, #1c2541, #3a506b);
    color: #eaeaea;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, h4, h5 {
    color: #5bc0be;
    text-align: center;
    font-weight: 700;
}
.stTextArea textarea {
    background-color: #1c2541 !important;
    color: #eaeaea !important;
    border-radius: 10px;
    border: 1px solid #5bc0be;
    font-size: 16px;
}
div.stButton > button {
    background: linear-gradient(90deg, #45b7aa, #6fffe9);
    color: #0b132b;
    font-weight: bold;
    border-radius: 50px;
    height: 3.2em;
    width: 60%;
    border: none;
    box-shadow: 0 0 20px #5bc0be;
    margin: 25px auto;
    display: block;
    letter-spacing: 1px;
    transition: all 0.3s ease-in-out;
}
div.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0 0 25px #6fffe9, 0 0 50px #5bc0be;
    background: linear-gradient(90deg, #6fffe9, #45b7aa);
}
.result-box {
    background-color: #1c2541;
    border: 1px solid #5bc0be;
    border-radius: 10px;
    padding: 15px;
    margin-top: 15px;
    font-size: 17px;
    color: #f8f8f8;
}
.footer {
    text-align: center;
    color: #9fa9b0;
    font-size: 13px;
    margin-top: 20px;
    padding-top: 10px;
    border-top: 1px solid #5bc0be;
}
.status {
    text-align: center;
    font-size: 15px;
    color: #6fffe9;
    font-weight: 500;
}
.download-buttons {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1>🧠 Bangla Zero-Shot Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<h4>Lightweight Version for Streamlit Cloud</h4>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_name = "csebuetnlp/mT5_m2m_small"  # ✅ lightweight version
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()
gpu_status = f"🟢 GPU Active: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "⚪ CPU Mode Active"
st.markdown(f"<p class='status'>{gpu_status}</p>", unsafe_allow_html=True)

# ---------- INPUT ----------
st.markdown("### ✏️ Enter Bangla Text to Summarize")
text = st.text_area(
    "Input Bangla Text",
    height=200,
    placeholder="এখানে একটি বাংলা অনুচ্ছেদ লিখুন...",
    label_visibility="collapsed"
)

# ---------- PDF CREATOR ----------
def create_pdf(summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    try:
        pdf.multi_cell(0, 10, txt="Bangla Summary:\n\n" + summary_text)
    except UnicodeEncodeError:
        pdf.multi_cell(0, 10, txt="Bangla Summary:\n\n" + summary_text.encode("utf-8", "ignore").decode("utf-8"))
    pdf_bytes = pdf.output(dest='S').encode('utf-8', 'ignore')
    return BytesIO(pdf_bytes)

# ---------- SAFE SUMMARIZATION ----------
def safe_generate(text):
    def run_generation():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_length=120,
                min_length=40,
                num_beams=2,
                repetition_penalty=1.1,
                early_stopping=True
            )
        return tokenizer.decode(ids[0], skip_special_tokens=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_generation)
        try:
            return future.result(timeout=60)  # 60 sec max
        except concurrent.futures.TimeoutError:
            return "⚠️ Summarization timed out. Please shorten your text."

# ---------- BUTTON ----------
summary = ""
if st.button("🚀 Generate Summary"):
    if not text.strip():
        st.warning("⚠️ অনুগ্রহ করে একটি বাংলা অনুচ্ছেদ লিখুন।")
    else:
        with st.spinner("Generating summary..."):
            summary = safe_generate(text)

        if summary.startswith("⚠️"):
            st.error(summary)
        else:
            st.markdown("<h3>📝 Generated Summary</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-box'>{summary}</div>", unsafe_allow_html=True)

            st.markdown("<div class='download-buttons'>", unsafe_allow_html=True)
            colA, colB = st.columns([1, 1])
            with colA:
                st.download_button(
                    "📄 Download as TXT",
                    data=summary.encode("utf-8"),
                    file_name="bangla_summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with colB:
                pdf_buffer = create_pdf(summary)
                st.download_button(
                    "🧾 Download as PDF",
                    data=pdf_buffer,
                    file_name="bangla_summary.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class='footer'>
Developed by <b>Your Name</b> | Thesis Visualization Project<br>
Model: csebuetnlp/mT5_m2m_small (optimized for CPU)
</div>
""", unsafe_allow_html=True)
