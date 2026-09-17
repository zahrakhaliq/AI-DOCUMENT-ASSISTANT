# AI Document Assistant

A simple Streamlit RAG application that lets you upload documents, search them using semantic and keyword search, and ask questions through Groq.

## Features

- PDF, DOCX, TXT and MD upload
- Separate extraction functions for each file type
- Filename and page metadata
- Text chunking with overlap
- Sentence Transformers embeddings
- FAISS vector search
- Simple keyword search
- Hybrid semantic + keyword retrieval
- Groq-powered question answering
- Answers are restricted to retrieved document context
- Retrieved sources shown after every answer
- Google Drive public file support
- Embeddings are created once when documents are processed
- Streamlit session state keeps the processed index available between questions
- Embedding model is cached with `st.cache_resource`

## Project structure

```text
ai-document-assistant/
├── app.py
├── requirements.txt
└── README.md
```

## 1. Install dependencies

Create a virtual environment if desired:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## 2. Add the Groq API key

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Do not put the key directly in `app.py`.

## 3. Run the app

```bash
streamlit run app.py
```

## How the pipeline works

### Step 1: Extraction

Each supported file type has its own extraction function:

- `extract_pdf()`
- `extract_docx()`
- `extract_txt()`
- `extract_md()`

PDF pages keep their page number. TXT, MD and DOCX do not have a reliable page number in this simple implementation, so their page value is stored as `None`.

### Step 2: Chunking

The extracted text is split into chunks of approximately 700 characters.

The chunks overlap by 100 characters so information near a chunk boundary is less likely to be lost.

Every chunk keeps:

```text
filename
page
chunk_number
text
```

### Step 3: Embeddings

Sentence Transformers converts every chunk into a vector.

The vectors are created only when the user clicks **Process Documents**.

They are stored in Streamlit session state:

```python
st.session_state.embeddings
```

The FAISS index is also stored:

```python
st.session_state.index
```

Therefore, asking another question does not recreate all document embeddings.

Only the new question is embedded.

### Step 4: FAISS search

The question is converted into an embedding and compared with the stored document vectors.

The most similar chunks become semantic-search candidates.

### Step 5: Keyword search

Important words are extracted from the question.

The app checks how many of those words occur in each chunk.

This gives every candidate a simple keyword score.

### Step 6: Hybrid search

The final score is:

```text
70% semantic similarity
30% keyword matching
```

The chunks are ranked using this combined score.

### Step 7: Groq

The top retrieved chunks are sent to Groq together with the user's question.

The prompt tells the model:

- use only the supplied context
- do not invent information
- say when the information is unavailable

### Step 8: Sources

After every answer, the app displays the retrieved chunks with:

- filename
- page number when available
- hybrid score
- semantic score
- keyword score
- retrieved text

## Google Drive

The starter implementation supports **publicly accessible Google Drive file links**.

Supported file types:

- PDF
- DOCX
- TXT
- MD

Paste the Drive file link into the sidebar and click **Process Documents**.

### About Drive folders

Reading an entire private Google Drive folder requires Google Drive API authentication, OAuth, or a service account.

This starter version deliberately keeps the implementation simple and does not add that authentication layer.

The rest of the pipeline is already shared: Drive files are downloaded and then pass through the same extraction, chunking, embedding and FAISS pipeline as local uploads.

## Important behavior

The app does not create embeddings every time you ask a question.

Document embeddings are created during processing:

```text
Documents
   ↓
Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS index
```

Questions use the existing index:

```text
Question
   ↓
Question embedding
   ↓
FAISS semantic search
   +
Keyword search
   ↓
Hybrid ranking
   ↓
Retrieved chunks
   ↓
Groq
   ↓
Answer + sources
```

## Simple limitations

This is intentionally a beginner-friendly implementation.

- Scanned/image-only PDFs need OCR, which is not included.
- DOCX page numbers are not available from normal paragraph extraction.
- Google Drive folder/private-file access needs authenticated Drive API integration.
- FAISS and embeddings are stored in Streamlit session state, so restarting the app requires processing the documents again.
- For a production application, a persistent vector database and document hashing/versioning would be appropriate.
