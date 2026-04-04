import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from components.embeddings import load_embeddings_and_db
from components.llm import load_llm, rewrite_and_generate_response
from components.config import SENTENCE_EMBEDDING_MODEL, FAISS_INDEX_PATH, LLM_MODEL_NAME, CROSS_ENCODER_MODEL, BM25_CORPUS_PATH
from components.dataLoader import load_json
from components.retriever import bm25_retriever, faiss_retriever
from components.reranker import rerank_and_select

embeddings, db = load_embeddings_and_db(model_name=SENTENCE_EMBEDDING_MODEL, index_path=FAISS_INDEX_PATH)

# Load BM25 corpus from JSON file
bm25_corpus = load_json(BM25_CORPUS_PATH)

bm25 = BM25Okapi([doc["text"].split() for doc in bm25_corpus])

# cross encoder for re-ranking
reranker = CrossEncoder(CROSS_ENCODER_MODEL)

st.set_page_config(
    page_title="Production Grade RAG Assistant",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Production Grade RAG Assistant")

llm = load_llm(model_name=LLM_MODEL_NAME)

col1, col2 = st.columns([3, 2])

with col1:
    query = st.text_input("Ask a question:")

    if query and llm:
        with st.spinner("Thinking..."):

            bm25_docs = bm25_retriever(query, bm25, bm25_corpus)
            vector_docs = faiss_retriever(query, db)
            top_docs = rerank_and_select(query, vector_docs, bm25_docs, reranker, top_k=5)

            context = ""
            sources = set()
            for d in top_docs:
                context += d.page_content if hasattr(d, 'page_content') else d["text"] + "\n\n"
                sources.add(d.metadata["source"] if hasattr(d, 'metadata') else d["metadata"]["source"])

            response = rewrite_and_generate_response(llm, query, context, st.session_state.chat_history)

        st.write("### Answer")
        st.write(response)

        st.write("### Sources")
        for s in sources:
            st.markdown(f"- [{s}]({s})")

        st.session_state.chat_history.append((query, response))

with col2:
    st.write("### Instructions")
    st.write("""
    - Enter a question related to the PDF content.
    - The assistant will use the most relevant sections of the document to answer.
    - Ensure your question is clear and specific for better results.
    """)

with st.sidebar:
    st.write("### Chat History")
    if st.session_state.chat_history:
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            preview = a[:100] + "..." if len(a) > 100 else a
            with st.expander(f"Assistant: {preview}"):
                st.write(a)
    else:
        st.write("No conversation yet.")
