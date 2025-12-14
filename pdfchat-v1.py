# =========================================================
# AI PDF Question Answering System (RAG + LIVE STREAMING)
# Single File | AIML Major Project
# =========================================================

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import requests
import json

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------- CONFIG ----------------
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "falcon:7b"     # try "mistral" for faster response
TOP_K = 2
MAX_CONTEXT_CHARS = 1500
OLLAMA_URL = "http://localhost:11434/api/generate"


# ---------------- STREAMLIT SETUP ----------------
st.set_page_config("AI PDF Chat", layout="wide")
st.title("📘 AI PDF Question Answering System")
st.caption("RAG-based | Live Streaming | Offline LLM")


# ---------------- LOAD EMBEDDER ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

embedder = load_embedder()


# ---------------- PDF TEXT EXTRACTION ----------------
def extract_pdf_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


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
        raise ValueError("PDF has no readable text (possibly scanned).")

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
    return context[:MAX_CONTEXT_CHARS]


# ---------------- STREAMING LLM CALL ----------------
def stream_llm_answer(context, question, output_box):
    prompt = f"""
You are an intelligent academic assistant.

INSTRUCTIONS:
- Use the context ONLY as a knowledge source
- DO NOT copy sentences verbatim
- Understand the meaning and explain in your own words
- If the answer is partially present, reason and infer carefully
- If not present, clearly say: "Not found in the document"

Answer Style:
- Clear
- Human-like
- Structured
- Concise but meaningful

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
            # "temperature": 0.2,
            # "num_predict": 200
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
uploaded_pdf = st.file_uploader("📂 Upload a PDF", type=["pdf"])

if uploaded_pdf:
    with st.spinner("📄 Reading PDF..."):
        raw_text = extract_pdf_text(uploaded_pdf)

    if len(raw_text.strip()) < 50:
        st.error("❌ This PDF has no readable text (likely scanned).")
        st.stop()

    with st.spinner("✂️ Splitting text..."):
        chunks = chunk_text(raw_text)

    with st.spinner("🧠 Creating vector database..."):
        index, stored_chunks = build_vector_store(chunks)

    st.success("✅ PDF processed successfully")

    st.divider()
    question = st.text_input("❓ Ask a question from this PDF")

    if question:
        with st.spinner("🔍 Retrieving relevant content..."):
            context = retrieve_context(question, index, stored_chunks)

        st.subheader("🤖 Answer (Live)")
        answer_box = st.empty()

        try:
            stream_llm_answer(context, question, answer_box)
        except requests.exceptions.Timeout:
            st.error("❌ LLM timed out. Try a shorter question.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

        with st.expander("🔎 Retrieved Context"):
            st.write(context)

else:
    st.info("⬆️ Upload a PDF to begin")
