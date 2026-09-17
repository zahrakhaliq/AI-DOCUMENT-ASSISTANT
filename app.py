import io
import os
import re
import pickle
from pathlib import Path

import faiss
import numpy as np
import requests
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="AI Document Assistant", page_icon="🔷", layout="wide")

CUSTOM_CSS = """
<style>
:root{
    --bg-app:#060b16;
    --bg-panel:#0d1b2e;
    --bg-panel-alt:#122544;
    --border:#1e3a5f;
    --border-soft:#16283f;
    --accent:#2f6fed;
    --accent-bright:#4c8dff;
    --accent-soft:rgba(47,111,237,0.14);
    --text-main:#e7edf7;
    --text-dim:#8fa3c4;
    --text-faint:#5f7594;
    --radius:14px;
}

.stApp{
    background:
        radial-gradient(circle at 15% 0%, #0e1f3a 0%, transparent 45%),
        radial-gradient(circle at 85% 10%, #0a1830 0%, transparent 40%),
        var(--bg-app);
    color:var(--text-main);
}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0a1628 0%, #081120 100%);
    border-right:1px solid var(--border-soft);
}
section[data-testid="stSidebar"] .block-container{ padding-top:1.6rem; }

h1,h2,h3,h4,h5{ color:var(--text-main) !important; letter-spacing:-0.01em; }
p, li, span, label, .stMarkdown{ color:var(--text-dim); }

/* ---------- Hero header ---------- */
.hero{
    background:linear-gradient(135deg, var(--bg-panel-alt) 0%, var(--bg-panel) 100%);
    border:1px solid var(--border);
    border-radius:var(--radius);
    padding:28px 32px;
    margin-bottom:22px;
    box-shadow:0 8px 30px rgba(0,0,0,0.35);
}
.hero h1{ font-size:1.9rem; margin:0 0 6px 0; }
.hero p{ margin:0; color:var(--text-dim); font-size:0.98rem; }
.hero .badge-row{ margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }

/* ---------- Generic pill / badge ---------- */
.pill{
    display:inline-flex; align-items:center; gap:6px;
    background:var(--accent-soft);
    border:1px solid rgba(76,141,255,0.35);
    color:#bcd4ff;
    padding:4px 12px;
    border-radius:999px;
    font-size:0.78rem;
    font-weight:500;
}
.pill.muted{
    background:rgba(255,255,255,0.04);
    border-color:var(--border);
    color:var(--text-dim);
}
.pill.score{
    background:rgba(47,111,237,0.10);
    border-color:var(--border);
    color:#a9c2ec;
    font-variant-numeric:tabular-nums;
}

/* ---------- Sidebar section title ---------- */
.sidebar-title{
    color:var(--text-main);
    font-weight:600;
    font-size:0.95rem;
    margin:4px 0 10px 0;
    display:flex; align-items:center; gap:8px;
}

/* ---------- File uploader dropzone ---------- */
[data-testid="stFileUploaderDropzone"]{
    background:var(--bg-panel) !important;
    border:1.5px dashed var(--border) !important;
    border-radius:12px !important;
}
[data-testid="stFileUploaderDropzone"]:hover{
    border-color:var(--accent-bright) !important;
}
[data-testid="stFileUploaderDropzone"] *{ color:var(--text-dim) !important; }

/* ---------- Text inputs ---------- */
.stTextInput input, .stTextArea textarea{
    background:var(--bg-panel) !important;
    border:1px solid var(--border) !important;
    color:var(--text-main) !important;
    border-radius:10px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus{
    border-color:var(--accent-bright) !important;
    box-shadow:0 0 0 1px var(--accent-bright) !important;
}
.stTextInput input::placeholder{ color:var(--text-faint) !important; }

/* ---------- Buttons ---------- */
.stButton>button{
    background:linear-gradient(135deg, var(--accent) 0%, #1d4fc4 100%) !important;
    color:#ffffff !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    border-radius:10px !important;
    font-weight:600 !important;
    padding:0.55rem 1.1rem !important;
    box-shadow:0 4px 14px rgba(47,111,237,0.25);
    transition:filter 0.15s ease, transform 0.05s ease;
}
.stButton>button:hover{ filter:brightness(1.12); }
.stButton>button:active{ transform:translateY(1px); }
.stButton>button[kind="secondary"]{
    background:var(--bg-panel-alt) !important;
    box-shadow:none;
    border:1px solid var(--border) !important;
}

/* ---------- Cards (st.container(border=True)) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
    background:var(--bg-panel);
    border:1px solid var(--border) !important;
    border-radius:var(--radius) !important;
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"]{
    background:var(--bg-panel-alt);
    border:1px solid var(--border);
    border-radius:12px;
    padding:12px 16px;
}
[data-testid="stMetricValue"]{ color:var(--accent-bright) !important; }
[data-testid="stMetricLabel"]{ color:var(--text-dim) !important; }

/* ---------- Expander ---------- */
[data-testid="stExpander"]{
    background:var(--bg-panel-alt);
    border:1px solid var(--border) !important;
    border-radius:12px !important;
    overflow:hidden;
}
[data-testid="stExpander"] summary{ color:var(--text-main) !important; }

/* ---------- Chat messages ---------- */
[data-testid="stChatMessage"]{
    background:var(--bg-panel);
    border:1px solid var(--border);
    border-radius:14px;
    padding:4px 6px;
    margin-bottom:10px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
    background:linear-gradient(135deg,#132a4d 0%, #0d1b2e 100%);
    border-color:rgba(76,141,255,0.30);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]){
    background:var(--bg-panel);
    border-left:3px solid var(--accent-bright);
}

/* ---------- Divider ---------- */
hr{ border-color:var(--border-soft) !important; }

/* ---------- Alerts ---------- */
[data-testid="stAlert"]{
    border-radius:12px;
    border:1px solid var(--border);
}

/* ---------- File chip list ---------- */
.file-chip{
    display:flex; align-items:center; gap:8px;
    background:var(--bg-panel-alt);
    border:1px solid var(--border);
    border-radius:10px;
    padding:8px 12px;
    margin-bottom:6px;
    font-size:0.88rem;
    color:var(--text-main);
}
.file-chip .dot{
    width:7px; height:7px; border-radius:50%;
    background:var(--accent-bright);
    box-shadow:0 0 6px var(--accent-bright);
    flex-shrink:0;
}

/* ---------- Source card ---------- */
.source-card{
    background:var(--bg-panel-alt);
    border:1px solid var(--border);
    border-radius:12px;
    padding:12px 14px;
    margin-bottom:10px;
}
.source-card .source-head{
    display:flex; justify-content:space-between; align-items:center;
    margin-bottom:8px; flex-wrap:wrap; gap:6px;
}
.source-card .source-name{ color:var(--text-main); font-weight:600; font-size:0.86rem; }
.source-card .source-text{
    color:var(--text-dim); font-size:0.85rem; line-height:1.5;
    max-height:120px; overflow-y:auto;
}

section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{ background:#0a1628 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 5

# -----------------------------
# Session state
# -----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "index" not in st.session_state:
    st.session_state.index = None
if "documents_key" not in st.session_state:
    st.session_state.documents_key = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


# -----------------------------
# Text extraction
# -----------------------------
def extract_pdf(file_bytes, filename):
    reader = PdfReader(io.BytesIO(file_bytes))
    results = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            results.append({
                "filename": filename,
                "page": page_number,
                "text": text.strip()
            })

    return results


def extract_docx(file_bytes, filename):
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(
        paragraph.text for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return [{
        "filename": filename,
        "page": None,
        "text": text.strip()
    }] if text.strip() else []


def extract_txt(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore")

    return [{
        "filename": filename,
        "page": None,
        "text": text.strip()
    }] if text.strip() else []


def extract_md(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore")

    return [{
        "filename": filename,
        "page": None,
        "text": text.strip()
    }] if text.strip() else []


def extract_document(file_bytes, filename):
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)
    if extension == ".docx":
        return extract_docx(file_bytes, filename)
    if extension == ".txt":
        return extract_txt(file_bytes, filename)
    if extension == ".md":
        return extract_md(file_bytes, filename)

    return []


# -----------------------------
# Chunking
# -----------------------------
def create_chunks(extracted_pages):
    chunks = []

    for item in extracted_pages:
        text = item["text"]
        start = 0
        chunk_number = 1

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "filename": item["filename"],
                    "page": item["page"],
                    "chunk_number": chunk_number,
                    "text": chunk_text
                })

            if end >= len(text):
                break

            start = end - CHUNK_OVERLAP
            chunk_number += 1

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def build_vector_index(chunks):
    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return embeddings, index


# -----------------------------
# Keyword search
# -----------------------------
def important_words(question):
    words = re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "what", "who",
        "when", "where", "why", "how", "can", "could", "would", "should",
        "does", "do", "did", "of", "to", "in", "on", "for", "and", "or",
        "with", "about", "from", "this", "that", "it", "i", "me", "my"
    }

    return [word for word in words if word not in stop_words and len(word) > 2]


def keyword_score(question, text):
    words = important_words(question)
    if not words:
        return 0.0

    text_lower = text.lower()
    matches = sum(1 for word in words if word in text_lower)

    return matches / len(words)


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, top_k=TOP_K):
    if not st.session_state.chunks or st.session_state.index is None:
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    semantic_scores, semantic_ids = st.session_state.index.search(
        question_embedding,
        min(top_k * 3, len(st.session_state.chunks))
    )

    candidates = {}

    for score, idx in zip(semantic_scores[0], semantic_ids[0]):
        if idx >= 0:
            candidates[int(idx)] = float(score)

    # Also consider every chunk for keyword matching.
    # This keeps the keyword part simple and useful for smaller documents.
    for idx, chunk in enumerate(st.session_state.chunks):
        score = keyword_score(question, chunk["text"])
        if score > 0:
            candidates[idx] = max(candidates.get(idx, 0), 0)

    ranked = []

    for idx in candidates:
        semantic = candidates.get(idx, 0.0)
        keyword = keyword_score(
            question,
            st.session_state.chunks[idx]["text"]
        )

        # 70% semantic + 30% keyword
        hybrid = (0.7 * semantic) + (0.3 * keyword)

        ranked.append({
            "score": hybrid,
            "semantic_score": semantic,
            "keyword_score": keyword,
            "chunk": st.session_state.chunks[idx]
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    return ranked[:top_k]


# -----------------------------
# Groq
# -----------------------------
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

    if not api_key:
        return None

    return Groq(api_key=api_key)


def answer_question(question, retrieved_chunks):
    client = get_groq_client()

    if client is None:
        return "Please add GROQ_API_KEY to Streamlit secrets."

    context_parts = []

    for i, result in enumerate(retrieved_chunks, start=1):
        chunk = result["chunk"]
        page_text = f"Page {chunk['page']}" if chunk["page"] else "Page not available"

        context_parts.append(
            f"[Source {i}]\n"
            f"Filename: {chunk['filename']}\n"
            f"{page_text}\n"
            f"Text:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
Answer the user's question using ONLY the document context below.

If the answer is not available in the provided context, say:
"The information is not available in the provided documents."

Do not invent facts. Do not use outside knowledge.
Give a clear and concise answer.

USER QUESTION:
{question}

DOCUMENT CONTEXT:
{context}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a document question-answering assistant. Use only the supplied context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# -----------------------------
# Google Drive
# -----------------------------
def extract_drive_id(url):
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/folders/([a-zA-Z0-9_-]+)",
        r"/document/d/([a-zA-Z0-9_-]+)",
        r"/spreadsheets/d/([a-zA-Z0-9_-]+)",
        r"/presentation/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def is_native_google_doc(url):
    """
    True for Google Docs/Sheets/Slides links, which are not binary files
    in Drive and cannot be downloaded via the uc?export=download endpoint.
    They must be exported instead.
    """
    if "/document/d/" in url:
        return "docx"
    if "/spreadsheets/d/" in url:
        return "xlsx"
    if "/presentation/d/" in url:
        return "pptx"
    return None


def download_drive_file(file_id, filename):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    return response.content, filename


def load_drive_file(url):
    """
    Supports publicly accessible Google Drive FILE links.

    A private Drive file/folder requires authentication and is intentionally
    not handled here to keep this starter app simple.
    """
    file_id = extract_drive_id(url)

    if not file_id:
        raise ValueError("Could not find a Google Drive file ID in the link.")

    # Google Drive folder links are not directly downloadable with this
    # simple public-link approach.
    if "/folders/" in url:
        raise ValueError(
            "Folder links require Google Drive API authentication. "
            "Use a public file link here, or extend the app with OAuth/service-account access."
        )

    native_format = is_native_google_doc(url)

    if native_format:
        # Google Docs/Sheets/Slides are not binary files in Drive, so they
        # must be exported to a real file format instead of downloaded
        # with uc?export=download.
        filename = f"drive_file_{file_id}.{native_format}"

        response = requests.get(
            f"https://docs.google.com/{'document' if native_format == 'docx' else 'spreadsheets' if native_format == 'xlsx' else 'presentation'}/d/{file_id}/export?format={native_format}",
            timeout=30
        )
        response.raise_for_status()

        return response.content, filename

    filename = f"drive_file_{file_id}"

    response = requests.get(
        f"https://drive.google.com/uc?export=download&id={file_id}",
        timeout=30
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    # Try to detect an extension from the response.
    if "pdf" in content_type:
        filename += ".pdf"
    elif "word" in content_type or "officedocument" in content_type:
        filename += ".docx"
    elif "text" in content_type:
        filename += ".txt"
    else:
        raise ValueError(
            "The Drive file type could not be detected. "
            "Use a PDF, DOCX, TXT, or MD file."
        )

    return response.content, filename


# -----------------------------
# Document processing
# -----------------------------
def process_documents(files):
    all_extracted = []

    for file in files:
        all_extracted.extend(
            extract_document(file["bytes"], file["filename"])
        )

    chunks = create_chunks(all_extracted)

    if not chunks:
        raise ValueError("No readable text was found in the documents.")

    embeddings, index = build_vector_index(chunks)

    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.index = index

    return all_extracted, chunks


# -----------------------------
# UI
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🔷 AI Document Assistant</h1>
        <p>Upload documents or load a public Google Drive file, then ask questions
        using hybrid semantic + keyword retrieval, answered strictly from your sources.</p>
        <div class="badge-row">
            <span class="pill">Semantic Search</span>
            <span class="pill">Keyword Search</span>
            <span class="pill">Hybrid Ranking</span>
            <span class="pill muted">Groq-Powered</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-title">📂 Document Sources</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True
    )

    st.markdown('<div class="sidebar-title">🔗 Google Drive</div>', unsafe_allow_html=True)

    drive_link = st.text_input(
        "Google Drive file link",
        placeholder="Paste a public Drive file link",
        label_visibility="collapsed",
    )

    st.write("")
    process_button = st.button("⚙️  Process Documents", type="primary", use_container_width=True)

    if process_button:
        files_to_process = []

        for uploaded in uploaded_files or []:
            files_to_process.append({
                "filename": uploaded.name,
                "bytes": uploaded.getvalue()
            })

        if drive_link.strip():
            try:
                drive_bytes, drive_filename = load_drive_file(drive_link.strip())
                files_to_process.append({
                    "filename": drive_filename,
                    "bytes": drive_bytes
                })
            except Exception as error:
                st.error(f"Google Drive error: {error}")

        if files_to_process:
            try:
                with st.spinner("Extracting text, creating chunks and embeddings..."):
                    extracted, chunks = process_documents(files_to_process)

                st.session_state.documents_key = tuple(
                    item["filename"] for item in files_to_process
                )

                st.success(
                    f"Processed {len(files_to_process)} document(s) and created "
                    f"{len(chunks)} chunks."
                )
            except Exception as error:
                st.error(f"Processing error: {error}")
        else:
            st.warning("Please upload a document or provide a Google Drive file link.")

    if st.session_state.qa_history:
        st.write("")
        if st.button("🗑️  Clear conversation", type="secondary", use_container_width=True):
            st.session_state.qa_history = []
            st.rerun()


if st.session_state.chunks:
    with st.container(border=True):
        st.markdown("#### 📊 Document Information")

        filenames = sorted({
            chunk["filename"] for chunk in st.session_state.chunks
        })

        col1, col2 = st.columns(2)
        col1.metric("Documents", len(filenames))
        col2.metric("Chunks", len(st.session_state.chunks))

        st.write("")
        st.markdown("**Loaded files**")
        for filename in filenames:
            st.markdown(
                f'<div class="file-chip"><span class="dot"></span>{filename}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("View extracted document chunks"):
            for i, chunk in enumerate(st.session_state.chunks, start=1):
                page = (
                    f"Page {chunk['page']}"
                    if chunk["page"]
                    else "Page not available"
                )

                st.markdown(
                    f"**Chunk {i} — {chunk['filename']} — {page}**"
                )
                st.write(chunk["text"])

    st.write("")


st.markdown("#### 💬 Ask a Question")

for turn in st.session_state.qa_history:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(turn["question"])

    with st.chat_message("assistant", avatar="🔷"):
        st.write(turn["answer"])

        results = turn["results"]
        if results:
            with st.expander(f"📎 {len(results)} retrieved source(s)"):
                for i, result in enumerate(results, start=1):
                    chunk = result["chunk"]
                    page = (
                        f"Page {chunk['page']}"
                        if chunk["page"]
                        else "Page not available"
                    )

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-head">
                                <span class="source-name">Source {i} · {chunk['filename']} — {page}</span>
                            </div>
                            <div>
                                <span class="pill score">Hybrid {result['score']:.3f}</span>
                                <span class="pill score">Semantic {result['semantic_score']:.3f}</span>
                                <span class="pill score">Keyword {result['keyword_score']:.3f}</span>
                            </div>
                            <br/>
                            <div class="source-text">{chunk['text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

question = st.chat_input("Ask something about your documents...")

if question:
    if not st.session_state.chunks:
        st.warning("Please process at least one document first.")
    else:
        with st.spinner("Searching documents and generating an answer..."):
            results = hybrid_search(question)
            answer = answer_question(question, results)

        st.session_state.qa_history.append({
            "question": question,
            "answer": answer,
            "results": results,
        })
        st.rerun()

if not st.session_state.chunks:
    st.info("Upload a document or add a Google Drive file, then click Process Documents.")
elif not st.session_state.qa_history:
    st.info("Ask your first question using the box below.")
