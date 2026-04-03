import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

docs = []

folder = "documents"

for file in os.listdir(folder):
    if file.endswith(".pdf"):
        file_path = os.path.join(folder, file)
        print(f"Loading document: {file_path}")

        loader = PyPDFLoader(file_path)
        loaded_pages = loader.load()

        for page in loaded_pages:
            page.metadata["source"] = file

        docs.extend(loaded_pages)

print("Total documents loaded: ", len(docs))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)
print("Chunks created: ", len(chunks))

# save chunks to json file for BM25 retrieval
# all_chunks = [doc.page_content for doc in chunks]
bm25_data = [
    {
        "text": doc.page_content,
        "metadata": doc.metadata
    }
    for doc in chunks
]

with open("data/bm25_corpus.json", "w") as f:
    json.dump(bm25_data, f)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print("Index saved locally.")