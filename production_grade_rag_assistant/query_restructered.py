from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from components.embeddings import load_embeddings_and_db
from components.llm import load_llm, rewrite_and_generate_response
from components.config import SENTENCE_EMBEDDING_MODEL, FAISS_INDEX_PATH, LLM_MODEL_NAME, CROSS_ENCODER_MODEL, BM25_CORPUS_PATH
from components.dataLoader import load_json
from components.retriever import bm25_retriever, faiss_retriever
from components.reranker import rerank_and_select

# Load FAISS index and embeddings
embeddings, db = load_embeddings_and_db(model_name=SENTENCE_EMBEDDING_MODEL, index_path=FAISS_INDEX_PATH)

# Load BM25 corpus from JSON file
bm25_corpus = load_json(BM25_CORPUS_PATH)

bm25 = BM25Okapi([doc["text"].split() for doc in bm25_corpus])

# cross encoder for re-ranking
reranker = CrossEncoder(CROSS_ENCODER_MODEL)

print("\n---Production Grade RAG Assistant---")

chat_history = []

# llm for query rewriting and response generation
llm = load_llm(model_name=LLM_MODEL_NAME)

while True:
    query = input("\nAsk (Type 'q' to quit): ")

    if(query.lower() == 'q'):
        break

    # BM25 retrieval
    bm25_docs = bm25_retriever(query, bm25, bm25_corpus)
    
    # Vector-based retrieval using FAISS
    vector_docs = faiss_retriever(query, db)

    # score documents using cross-encoder re-ranker
    top_docs = rerank_and_select(query, vector_docs, bm25_docs, reranker, top_k=5)

    context = ""
    sources = set()

    for d in top_docs:
        context += d.page_content if hasattr(d, 'page_content') else d["text"] + "\n\n"
        sources.add(d.metadata["source"] if hasattr(d, 'metadata') else d["metadata"]["source"])

    response = rewrite_and_generate_response(llm, query, context, chat_history)

    print("\nAnswer: ", response)
    print("\nSources: ")
    for s in sources:
        print("- ", s)

    chat_history.append((query, response))

print("\n---Production Grade RAG Assistant Stopped---")