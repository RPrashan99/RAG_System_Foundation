import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

docs = []

#Load PDF document dynamically from the "documents" directory
folder = "documents"
44
for file in os.listdir(folder):
    if file.endswith(".pdf"):
        file_path = os.path.join(folder, file)
        print(f"Loading document: {file_path}")

        loader = PyPDFLoader(file_path)
        loaded_pages = loader.load()

        for page in loaded_pages:
            page.metadata["source"] = file # To identify the source document for each page

        docs.extend(loaded_pages)

print("Total documents loaded: ", len(docs))

#Split text to chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)
print("Chunks created: ", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print("Index saved locally.")