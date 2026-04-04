def score_documents(query, candidate_docs, reranker):
    pairs = [(query, doc.page_content) for doc in candidate_docs]
    scores = reranker.predict(pairs)
    return scores

def score_documents_with_bm25(query, vector_docs, bm25_docs, reranker):
    pairs = [(query, doc.page_content) for doc in vector_docs]
    pairs += [(query, doc["text"]) for doc in bm25_docs]
    scores = reranker.predict(pairs)
    return scores

def rerank_documents(candidate_docs, scores):
    ranked_docs = [
        doc for _, doc in sorted(
            zip(scores, candidate_docs), 
            key=lambda x: x[0], 
            reverse=True)
    ]
    return ranked_docs

def select_top_documents(ranked_docs, top_k=5):
    return ranked_docs[:top_k]

def rerank_and_select(query, vector_docs, bm25_docs, reranker, top_k=5):
    scores = score_documents_with_bm25(query, vector_docs, bm25_docs, reranker)
    ranked_docs = rerank_documents(vector_docs + bm25_docs, scores)
    top_docs = select_top_documents(ranked_docs, top_k)
    return top_docs