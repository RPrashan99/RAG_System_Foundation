# RAG PDF Assistants

This workspace contains multiple related projects that implement **Retrieval-Augmented Generation (RAG)** over PDF documents using **FAISS** for vector search and **Ollama** for local LLM inference.

- **`multi_document_rag_assistant/`** – A Streamlit-based app that lets you search across multiple PDFs at once.
- **`rag-pdf-assistant/`** – A simpler RAG demo focused on a single PDF with both a Streamlit UI and a terminal query client.

---

## ⚡ Quick Start (Common)

### 1) Install dependencies

Each project has its own requirements. Create and activate a Python virtual environment, then install dependencies for the project you want to run.

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2) Start Ollama (local LLM)

Both projects rely on a locally running Ollama server.

```bash
ollama server
```

Make sure you have a model (e.g. `llama3`) installed and available in Ollama.

---

## 📦 Projects Overview

### 1) `multi_document_rag_assistant/`

- **Purpose:** Search across multiple PDFs stored in `documents/`.
- **Indexing:** Run `ingest.py` to load all PDF files in `documents/`, chunk them, embed them, and save a FAISS index in `faiss_index/`.
- **Query:** Run `app.py` to start the Streamlit UI.

### 2) `rag-pdf-assistant/`

- **Purpose:** A simpler RAG demo for a specific PDF (configured in `data/`).
- **Indexing:** Run `ingest.py` to build a FAISS index from the PDF in `data/`.
- **Query (UI):** Run `streamlit run app.py`.
- **Query (terminal):** Run `python query.py`.
- **Evaluation:** `test/evaluate_rag.py` can run evaluation metrics (requires extra dependencies including `datasets` and `ragas` and high accuracy model).

---

## 🔍 Notes

- The projects use **sentence-transformers/all-MiniLM-L6-v2** for embeddings.
- The FAISS indexes are stored locally under `faiss_index/` in each project.
- Make sure PDF files are placed in the correct folder before running ingestion.

---

## 📌 Tips

- If you make changes to the PDF set, re-run `ingest.py` to refresh the index.
- If the app cannot connect to Ollama, confirm the server is running and the model name matches (`llama3`).
