# app.py
# StudyMate — Multi-PDF Q&A with persistent conversation + per-PDF notes

import os
import re
import io
import datetime
from typing import List, Tuple
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import ollama

try:
    from streamlit_quill import st_quill
    QUILL_AVAILABLE = True
except ImportError:
    QUILL_AVAILABLE = False

try:
    import faiss
    FAISS = True
except ImportError:
    FAISS = False

MODEL_NAME = "granite3.3:8b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NOTES_DIR = "notes"
os.makedirs(NOTES_DIR, exist_ok=True)

st.set_page_config(page_title="StudyMate — Multi-PDF Q&A", layout="wide")
st.title("📚 StudyMate — Multi-PDF Q&A via Ollama")

with st.sidebar:
    st.header("⚙️ Settings")
    CHUNK_WORDS = st.number_input("Chunk size (words)", 100, 1200, 350, step=25)
    CHUNK_OVERLAP = st.number_input("Overlap (words)", 0, 400, 60, step=10)
    TOP_K = st.slider("Top-K chunks", 1, 12, 5)
    MAX_NEW = st.slider("Max new tokens", 64, 1024, 350, step=32)
    TEMPERATURE = st.slider("Temperature", 0.0, 1.0, 0.2, step=0.05)

@st.cache_resource(show_spinner=True)
def get_embedder(model_name: str):
    return SentenceTransformer(model_name)

embedder = get_embedder(EMBED_MODEL)

def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def html_to_text(html_str: str) -> str:
    return BeautifulSoup(html_str, "html.parser").get_text("\n")

def extract_pdf_text(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    return "\n\n".join([p.extract_text() or "" for p in reader.pages])

def word_chunk(text: str, chunk_words: int, overlap: int) -> List[str]:
    words = clean_text(text).split()
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_words, n)
        chunks.append(" ".join(words[start:end]))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

class VectorStore:
    def __init__(self, dim: int, use_faiss: bool = True):
        self.use_faiss = use_faiss and FAISS
        self.dim = dim
        self.index = None
        self.embs = None

    def build(self, embs: np.ndarray):
        if self.use_faiss:
            faiss.normalize_L2(embs)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embs.astype("float32"))
        else:
            self.embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    def search(self, q_emb: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_faiss:
            faiss.normalize_L2(q_emb)
            scores, idxs = self.index.search(q_emb.astype("float32"), k)
            return idxs[0], scores[0]
        sims = (self.embs @ q_emb.T).squeeze(1)
        idxs = np.argsort(-sims)[:k]
        return idxs, sims[idxs]

def build_index(chunks: List[str]) -> VectorStore:
    embs = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    vs = VectorStore(dim=embs.shape[1], use_faiss=True)
    vs.build(embs)
    return vs

def build_prompt(question: str, context_chunks: List[str]) -> str:
    ctx = "\n\n".join(context_chunks)
    system = ("You are StudyMate, a study assistant. "
              "Answer ONLY using the provided context. "
              "If the answer is not in the context, say so.")
    return f"{system}\n\nQuestion: {question}\n\nContext:\n{ctx}\nAnswer:"

def call_ollama(prompt: str, max_new_tokens: int, temperature: float) -> str:
    try:
        r = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Helpful AI study assistant."},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": temperature,
                "num_predict": max_new_tokens
            }
        )
        return r['message']['content']
    except Exception as e:
        return f"❌ Ollama error: {e}"

def get_notes_file(pdf_name: str) -> str:
    base = os.path.splitext(os.path.basename(pdf_name))[0]
    return os.path.join(NOTES_DIR, f"{base}_notes.txt")

uploaded_pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_pdfs:
    if "pdf_data" not in st.session_state:
        st.session_state["pdf_data"] = {}

    for pdf in uploaded_pdfs:
        if pdf.name not in st.session_state["pdf_data"]:
            with st.spinner(f"Processing {pdf.name}..."):
                text = extract_pdf_text(pdf)
                chunks = word_chunk(text, CHUNK_WORDS, CHUNK_OVERLAP)
                vs = build_index(chunks)
                st.session_state["pdf_data"][pdf.name] = {
                    "text": text,
                    "chunks": chunks,
                    "vstore": vs,
                    "conversation": []  # persistent Q/A
                }

    mode = st.radio("Select Q&A mode", ["Single PDF", "Merged Knowledge Base"])
    if mode == "Single PDF":
        selected_pdf = st.selectbox("Select PDF for Q&A / Notes", list(st.session_state["pdf_data"].keys()))
    else:
        selected_pdf = None
else:
    selected_pdf = None
    mode = None

if mode == "Single PDF" and selected_pdf:
    st.subheader(f"📝 Notes for: {selected_pdf}")
    notes_file = get_notes_file(selected_pdf)

    if QUILL_AVAILABLE:
        notes_html = st_quill(
            value=st.session_state.get(f"notes_html_{selected_pdf}", ""),
            html=True,
            placeholder="Write your notes here..."
        )
    else:
        notes_html = st.text_area("Notes", value=st.session_state.get(f"notes_html_{selected_pdf}", ""))

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save Notes"):
            st.session_state[f"notes_html_{selected_pdf}"] = notes_html
            plain_text = html_to_text(notes_html) if QUILL_AVAILABLE else notes_html
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            conversation = st.session_state["pdf_data"][selected_pdf].get("conversation", [])
            qa_text = ""
            if conversation:
                qa_text = "\n\n".join(
                    [f"Q: {q}\nA: {a}" for q, a in conversation]
                )

            divider = "\n" + "="*50 + "\n"
            entry = f"{divider}📄 {selected_pdf} — 🕒 {timestamp}\n{plain_text.strip()}\n\n{qa_text.strip()}\n"
            with open(notes_file, "a", encoding="utf-8") as f:
                f.write(entry)
            st.success("Notes + AI Q/A saved!")

    with col2:
        if st.button("🧹 Clear Notes"):
            st.session_state[f"notes_html_{selected_pdf}"] = ""
            st.success("Cleared notes")

    with col3:
        if st.button("🗑️ Clear Conversation"):
            st.session_state["pdf_data"][selected_pdf]["conversation"] = []
            st.success("Conversation cleared")

    st.markdown("### 💬 Conversation History")
    for q, a in st.session_state["pdf_data"][selected_pdf]["conversation"]:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")

    st.markdown("### 📜 Saved Notes")
    for pdf_name in st.session_state["pdf_data"].keys():
        file_path = get_notes_file(pdf_name)
        if os.path.exists(file_path):
            with st.expander(f"📄 {pdf_name} — Saved Notes"):
                with open(file_path, "r", encoding="utf-8") as f:
                    saved = f.read().strip()
                st.text_area(f"Notes for {pdf_name}", saved, height=300)

if uploaded_pdfs:
    st.subheader("💬 Ask AI")
    question = st.text_input("Enter your question")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Enter a question")
        else:
            if mode == "Single PDF" and selected_pdf:
                chunks = st.session_state["pdf_data"][selected_pdf]["chunks"]
                vstore = st.session_state["pdf_data"][selected_pdf]["vstore"]
            else:
                all_chunks = []
                for data in st.session_state["pdf_data"].values():
                    all_chunks.extend(data["chunks"])
                merged_vs = build_index(all_chunks)
                chunks = all_chunks
                vstore = merged_vs

            q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
            idxs, sims = vstore.search(q_emb, k=TOP_K)
            top_chunks = [chunks[i] for i in idxs]
            prompt = build_prompt(question, top_chunks)
            with st.spinner("Querying Ollama..."):
                ans = call_ollama(prompt, max_new_tokens=MAX_NEW, temperature=TEMPERATURE)

            # store Q/A persistently
            if mode == "Single PDF" and selected_pdf:
                st.session_state["pdf_data"][selected_pdf]["conversation"].append((question, ans))

            st.rerun()  
