# =========================================================
# AI PDF Question Answering System
# RAG + LIVE STREAMING + OCR SUPPORT
# Single File | AIML Major Project
# =========================================================

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import pytesseract
from PIL import Image
import io
import os

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama


# ---------------- CONFIG ----------------
EMBED_MODEL = "all-MiniLM-L6-v2"
LLAMA_MODEL_PATH = "models/llama-3-8b-instruct.Q4_K_M.gguf"

TOP_K = 3
MAX_CONTEXT_CHARS = 1500
MAX_TOKENS = 300


# ---------------- STREAMLIT SETUP ----------------
st.set_page_config(
    page_title="AI PDF Chat with OCR",
    layout="wide"
)

st.title("📘 AI PDF Question Answering System")
st.caption("RAG-based | Live Streaming | OCR Enabled | Offline LLaMA (llama.cpp)")


# ---------------- LOAD EMBEDDER ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

embedder = load_embedder()


# ---------------- LOAD LLM ----------------
@st.cache_resource
def load_llm():
    if not os.path.exists(LLAMA_MODEL_PATH):
        raise FileNotFoundError(
            f"LLaMA model not found at {LLAMA_MODEL_PATH}. "
            "Download GGUF model during build or place it manually."
        )

    return Llama(
        model_path=LLAMA_MODEL_PATH,
        n_ctx=4096,
        n_threads=4,
        n_batch=512,
        verbose=False
    )

llm = load_llm()


# ---------------- OCR FROM IMAGE ----------------
def ocr_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pytesseract.image_to_string(image)


# ---------------- PDF TEXT EXTRACTION (TEXT + OCR) ----------------
def extract_pdf_text_with_ocr(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""

    for page in doc:
        page_text = page.get_text().strip()

        if page_text:
            full_text += page_text + "\n"
        else:
            images = page.get_images(full=True)
            for img in images:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                full_text += ocr_image(image_bytes) + "\n"

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
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, chunks


# ---------------- RETRIEVAL ----------------
def retrieve_context(question, index, chunks):
    q_emb = embedder.encode([question]).astype("float32")
    _, idxs = index.search(q_emb, min(TOP_K, len(chunks)))

    context = "\n".join(chunks[i] for i in idxs[0])
    context = "\n".join(dict.fromkeys(context.split("\n")))

    return context[:MAX_CONTEXT_CHARS]


# ---------------- STREAMING LLM ANSWER ----------------
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

    full_answer = ""

    for chunk in llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.3,
        top_p=0.9,
        stream=True,
        stop=["Context:", "Question:"]
    ):
        token = chunk["choices"][0]["text"]
        full_answer += token
        output_box.markdown(full_answer)

    return full_answer


# ---------------- UI ----------------
uploaded_pdf = st.file_uploader(
    "📂 Upload a PDF (Text or Scanned)",
    type=["pdf"]
)

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
        with st.spinner("🔍 Retrieving relevant context..."):
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
            st.error(f"❌ LLM Error: {e}")

        with st.expander("🔎 Retrieved Context"):
            st.write(context)

else:
    st.info("⬆️ Upload a PDF to begin")
