# Multi Document RAG Assistant

A Streamlit app to perform retrieval-augmented generation (RAG) over multiple PDF documents.

## 🔎 What it does

- Loads PDFs from `documents/`.
- Splits them into chunks.
- Embeds chunks using `sentence-transformers/all-MiniLM-L6-v2`.
- Stores embeddings in a local FAISS index (`faiss_index/`).
- Serves a Streamlit UI (`app.py`) that takes a user question, performs similarity search, and generates an answer with a local LLM (Ollama).

## 🚀 Setup

### 1) Create and activate a Python environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-ollama faiss-cpu pypdf python-dotenv streamlit
```

### 3) Start Ollama

```bash
ollama server
```

Confirm that a model like `llama3` is installed and available.

## 🧠 Build the index

Put your PDF files in the `documents/` directory, then run:

```bash
python ingest.py
```

This creates/updates the FAISS index under `faiss_index/`.

## 🧪 Run the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## 🗂️ Project Structure

- `documents/` — place your PDFs here.
- `faiss_index/` — generated FAISS index files.
- `ingest.py` — ingests PDFs, creates embeddings, builds index.
- `app.py` — Streamlit front-end for querying the index.

## 📝 Notes

- If you add/remove PDFs or update content, re-run `ingest.py` to refresh the index.
- The app displays the source filenames for the retrieved chunks.
