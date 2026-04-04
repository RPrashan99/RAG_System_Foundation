def faiss_retriever(query, db, k=6):
    vector_docs = db.similarity_search(query, k=k)
    return vector_docs

def bm25_retriever(query, bm25, bm25_corpus, k=6):
    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    top_n = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_docs = [bm25_corpus[i] for i in top_n]
    return bm25_docs