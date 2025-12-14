# =========================================================
# AI PDF Question Answering System
# RAG + LIVE STREAMING + OCR SUPPORT
# Single File | AIML Major Project
# =========================================================

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import requests
import json
import pytesseract
from PIL import Image
import io

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------- CONFIG ----------------
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "falcon:7b"     # better reasoning than llama3
TOP_K = 3
MAX_CONTEXT_CHARS = 1500
OLLAMA_URL = "http://localhost:11434/api/generate"


# ---------------- STREAMLIT SETUP ----------------
st.set_page_config("AI PDF Chat with OCR", layout="wide")
st.title("📘 AI PDF Question Answering System")
st.caption("RAG-based | Live Streaming | OCR Enabled | Offline AI")


# ---------------- LOAD EMBEDDER ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

embedder = load_embedder()


# ---------------- OCR FROM IMAGE ----------------
def ocr_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pytesseract.image_to_string(image)


# ---------------- PDF TEXT EXTRACTION (TEXT + OCR) ----------------
def extract_pdf_text_with_ocr(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""

    for page_num, page in enumerate(doc):
        page_text = page.get_text().strip()

        # If text exists, use it
        if page_text:
            full_text += page_text + "\n"
        else:
            # OCR fallback for scanned page
            images = page.get_images(full=True)
            for img in images:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ocr_text = ocr_image(image_bytes)
                full_text += ocr_text + "\n"

    return full_text


# ---------------- TEXT CHUNKING ----------------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)


# ---------------- VECTOR STORE ----------------
def build_vector_store(chunks):
    if not chunks:
        raise ValueError("No readable text found in PDF.")

    embeddings = embedder.encode(chunks)

    if len(embeddings.shape) == 1:
        embeddings = np.array([embeddings])

    embeddings = embeddings.astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, chunks


# ---------------- RETRIEVAL ----------------
def retrieve_context(question, index, chunks):
    q_emb = embedder.encode([question]).astype("float32")
    _, idxs = index.search(q_emb, min(TOP_K, len(chunks)))
    context = "\n".join(chunks[i] for i in idxs[0])

    # Deduplicate lines
    context = "\n".join(dict.fromkeys(context.split("\n")))
    return context[:MAX_CONTEXT_CHARS]


# ---------------- STREAMING LLM ----------------
def stream_llm_answer(context, question, output_box):
    prompt = f"""
You are an intelligent academic assistant.

INSTRUCTIONS:
- Use the context ONLY as a knowledge source
- Do NOT copy sentences verbatim
- Explain in your own words
- Be clear and structured
- If not found, say: "Not found in the document"

Context:
{context}

Question:
{question}

Final Answer:
"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.3,
            "num_predict": 250,
            "top_p": 0.9
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=30
    )

    full_answer = ""

    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            token = data.get("response", "")
            full_answer += token
            output_box.markdown(full_answer)

    return full_answer


# ---------------- UI ----------------
uploaded_pdf = st.file_uploader("📂 Upload a PDF (Text or Scanned)", type=["pdf"])

if uploaded_pdf:
    with st.spinner("📄 Extracting text (OCR enabled)..."):
        raw_text = extract_pdf_text_with_ocr(uploaded_pdf)

    if len(raw_text.strip()) < 50:
        st.error("❌ No readable text detected, even after OCR.")
        st.stop()

    with st.spinner("✂️ Chunking text..."):
        chunks = chunk_text(raw_text)

    with st.spinner("🧠 Building vector database..."):
        index, stored_chunks = build_vector_store(chunks)

    st.success("✅ PDF processed successfully (OCR applied if needed)")

    st.divider()
    question = st.text_input("❓ Ask a question from this PDF")

    if question:
        with st.spinner("🔍 Retrieving context..."):
            context = retrieve_context(
                f"In the context of the document, {question}",
                index,
                stored_chunks
            )

        st.subheader("🤖 Answer (Live)")
        answer_box = st.empty()

        try:
            stream_llm_answer(context, question, answer_box)
        except Exception as e:
            st.error(f"❌ Error: {e}")

        with st.expander("🔎 Retrieved Context"):
            st.write(context)

else:
    st.info("⬆️ Upload a PDF to begin")
