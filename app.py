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

st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")

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
st.title("📄 AI Document Assistant")
st.write(
    "Upload documents or load a public Google Drive file, then ask questions "
    "using semantic + keyword search."
)

st.sidebar.header("Document Sources")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, TXT or MD files",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True
)

drive_link = st.sidebar.text_input(
    "Google Drive file link",
    placeholder="Paste a public Drive file link"
)

process_button = st.sidebar.button("Process Documents", type="primary")

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


if st.session_state.chunks:
    st.subheader("Document Information")

    filenames = sorted({
        chunk["filename"] for chunk in st.session_state.chunks
    })

    col1, col2 = st.columns(2)
    col1.metric("Documents", len(filenames))
    col2.metric("Chunks", len(st.session_state.chunks))

    st.write("**Loaded files:**")
    for filename in filenames:
        st.write(f"• {filename}")

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


st.divider()

st.subheader("Ask a Question")

question = st.text_input(
    "What would you like to know?",
    placeholder="Ask something about your documents..."
)

if st.button("Ask", type="primary"):
    if not st.session_state.chunks:
        st.warning("Please process at least one document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating an answer..."):
            results = hybrid_search(question)
            answer = answer_question(question, results)

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Retrieved Sources")

        if not results:
            st.info("No relevant sources were found.")
        else:
            for i, result in enumerate(results, start=1):
                chunk = result["chunk"]
                page = (
                    f"Page {chunk['page']}"
                    if chunk["page"]
                    else "Page not available"
                )

                with st.expander(
                    f"Source {i}: {chunk['filename']} — {page}"
                ):
                    st.write(
                        f"**Hybrid score:** {result['score']:.3f}  \n"
                        f"**Semantic score:** {result['semantic_score']:.3f}  \n"
                        f"**Keyword score:** {result['keyword_score']:.3f}"
                    )
                    st.write(chunk["text"])
else:
    if not st.session_state.chunks:
        st.info("Upload a document or add a Google Drive file, then click Process Documents.")
