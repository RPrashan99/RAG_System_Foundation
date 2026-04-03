import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

embeddings  = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

with open("data/bm25_corpus.json", "r") as f:
    bm25_corpus = json.load(f)

bm25 = BM25Okapi([doc["text"].split() for doc in bm25_corpus])

# cross encoder for re-ranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("\n---Production Grade RAG Assistant---")

chat_history = []

try:
    llm = OllamaLLM(model="llama3")
except:
    print("\nModel not loaded. Check Ollama is running.")

while True:
    query = input("\nAsk (Type 'q' to quit): ")

    if(query.lower() == 'q'):
        break

    # BM25 retrieval
    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    top_n = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:6]

    # Keyword-based retrieval using BM25
    bm25_docs = [bm25_corpus[i] for i in top_n]
    
    # Vector-based retrieval using FAISS
    vector_docs = db.similarity_search(query, k=6)

    candidate_docs = vector_docs + bm25_docs

    # score documents using cross-encoder re-ranker
    pairs = [(query, doc.page_content) for doc in vector_docs]
    pairs += [(query, doc["text"]) for doc in bm25_docs]
    scores = reranker.predict(pairs)

    ranked_docs = [
        doc for _, doc in sorted(
            zip(scores, candidate_docs), 
            key=lambda x: x[0], 
            reverse=True)
    ]
    top_docs = ranked_docs[:5]

    context = ""
    sources = set()

    for d in top_docs:
        context += d.page_content if hasattr(d, 'page_content') else d["text"] + "\n\n"
        sources.add(d.metadata["source"] if hasattr(d, 'metadata') else d["metadata"]["source"])

    history_text = []

    for q, a in chat_history:
        history_text += f'User: {q}\nAssistant: {a}\n'


    # query rewrite for better retrieval
    rewrite_prompt = f"""
        Rewrite the user query so it is self-contained.

        conversation history:
        {history_text}

        User Question:
        {query}

        Rewritten Query:
    """

    rewritten_query = llm.invoke(rewrite_prompt).strip()
    print("\nRewritten Query: ", rewritten_query)

    prompt = f"""
        You are a helpful AI assistant.

        Use the conversation history and context to answer.

        Conversation History:
        {history_text}

        Context from documents:
        {context}

        User Question:
        {rewritten_query}

        Answer briefly and clearly using the context.
    """

    response = llm.invoke(prompt)
    print("\nAnswer: ", response)
    print("\nSources: ")
    for s in sources:
        print("- ", s)

    chat_history.append((query, response))

print("\n---Production Grade RAG Assistant Stopped---")