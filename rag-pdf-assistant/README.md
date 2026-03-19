# RAG PDF Assistant

A simple Retrieval-Augmented Generation (RAG) demo that uses a PDF document as the knowledge source.

## ✅ What it does

- Loads a PDF from `data/`.
- Splits it into chunks.
- Embeds each chunk using `sentence-transformers/all-MiniLM-L6-v2`.
- Stores the embeddings in a FAISS index under `faiss_index/`.
- Allows querying via:
  - A Streamlit UI (`app.py`)
  - A terminal prompt (`query.py`)

## ⚙️ Setup

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
pip install -r requirements.txt
```

### 3) Start Ollama

```bash
ollama server
```

Ensure a model such as `llama3` is installed.

## 📄 Build the index

Edit `data/Individual contribution.pdf` or replace it with your own PDF file (update the filename in `ingest.py`). Then run:

```bash
python ingest.py
```

## 🧪 Run

### Streamlit UI

```bash
streamlit run app.py
```

### Terminal query mode

```bash
python query.py
```

## 🧪 Evaluation (optional)

The `test/` folder contains an evaluation script that uses the `ragas` evaluation toolkit.

```bash
python test/evaluate_rag.py
```

## 📁 Project Layout

- `data/` — PDF used for the demo.
- `faiss_index/` — generated FAISS index.
- `ingest.py` — builds the FAISS index.
- `app.py` — Streamlit-based UI.
- `query.py` — terminal question/answer loop.
- `test/` — evaluation tools and dataset.
