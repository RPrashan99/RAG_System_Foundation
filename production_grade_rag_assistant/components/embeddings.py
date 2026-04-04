from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_embeddings_and_db(model_name="sentence-transformers/all-MiniLM-L6-v2", index_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return embeddings, db