# ==============================================================
# 🧠 Bangla Zero-Shot Summarizer (mT5_multilingual_XLSum)
# ✨ Glowing Animated Button + Download Options
# ==============================================================

import streamlit as st
import torch, time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from io import BytesIO
from fpdf import FPDF


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
    overflow-x: hidden;
}
h1, h2, h3, h4, h5 {
    color: #5bc0be;
    text-align: center;
    font-weight: 700;
}
hr {
    border: none;
    height: 2px;
    background: linear-gradient(to right, #5bc0be, #6fffe9);
    margin-top: 10px;
    margin-bottom: 20px;
}
.stTextArea textarea {
    background-color: #1c2541 !important;
    color: #eaeaea !important;
    border-radius: 10px;
    border: 1px solid #5bc0be;
    font-size: 16px;
}

/* 🔥 Custom glowing button styling */
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
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease-in-out;
    position: relative;
    overflow: hidden;
}

/* glowing hover animation */
div.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0 0 25px #6fffe9, 0 0 50px #5bc0be;
    background: linear-gradient(90deg, #6fffe9, #45b7aa);
}

/* animated glow border effect */
div.stButton > button::before {
    content: "";
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    border-radius: 50px;
    background: linear-gradient(45deg, #45b7aa, #6fffe9, #5bc0be, #45b7aa);
    background-size: 300% 300%;
    z-index: -1;
    animation: glowMove 4s linear infinite;
}
@keyframes glowMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.result-box {
    background-color: #1c2541;
    border: 1px solid #5bc0be;
    border-radius: 10px;
    padding: 15px;
    margin-top: 15px;
    font-size: 17px;
    color: #f8f8f8;
    animation: fadeIn 1s ease-in-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(15px); }
  to { opacity: 1; transform: translateY(0); }
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
.typing-cursor {
    display: inline-block;
    animation: blink 1s step-end infinite;
}
@keyframes blink {
    from, to { opacity: 0; }
    50% { opacity: 1; }
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
st.markdown("<h4>Model: csebuetnlp/mT5_multilingual_XLSum</h4>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
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
text = st.text_area("", height=200, placeholder="এখানে একটি বাংলা অনুচ্ছেদ লিখুন...")

# ---------- PDF CREATOR ----------
def create_pdf(summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Bangla Summary:\n\n" + summary_text)
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_bytes)


# ---------- SUMMARIZATION ----------
summary = ""
if st.button("🚀 Generate Summary"):
    if not text.strip():
        st.warning("⚠️ অনুগ্রহ করে একটি বাংলা অনুচ্ছেদ লিখুন।")
    else:
        with st.spinner("Generating summary..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                summary_ids = model.generate(
                    **inputs,
                    max_length=180,
                    min_length=60,
                    num_beams=6,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3,
                    early_stopping=True,
                )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # ---------- TYPING ANIMATION ----------
        st.markdown("<h3>📝 Generated Summary</h3>", unsafe_allow_html=True)
        placeholder = st.empty()
        typed_text = ""
        for char in summary:
            typed_text += char
            placeholder.markdown(
                f"<div class='result-box'>{typed_text}<span class='typing-cursor'>|</span></div>",
                unsafe_allow_html=True,
            )
            time.sleep(0.01)
        placeholder.markdown(f"<div class='result-box'>{summary}</div>", unsafe_allow_html=True)

        # ---------- DOWNLOAD OPTIONS ----------
        st.markdown("<div class='download-buttons'>", unsafe_allow_html=True)
        colA, colB = st.columns([1, 1])

        with colA:
            st.download_button(
                label="📄 Download as TXT",
                data=summary.encode("utf-8"),
                file_name="bangla_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with colB:
            pdf_buffer = create_pdf(summary)
            st.download_button(
                label="🧾 Download as PDF",
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
Model: csebuetnlp/mT5_multilingual_XLSum
</div>
""", unsafe_allow_html=True)
