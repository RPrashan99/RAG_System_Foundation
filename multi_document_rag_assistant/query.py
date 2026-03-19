from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

try:
    llm = OllamaLLM(model="llama3")
except:
    print("Model not available. Check ollama running.")

print("---Multi document RAG System---\n")

while True:
    query = input("\nAsk (Type 'q' to exit): ")

    if(query.lower() == "q"):
        break

    docs = db.similarity_search(query, k=6)

    context = ""
    sources = set()

    for d in docs:
        context += d.page_content + "\n\n"
        sources.add(d.metadata["source"])

    prompt = f"""
You are a helpful assistant.

Use ONLY the context below to answer.

Context:
{context}

Question:
{query}

Answer clearly.
"""
    
    response = llm.invoke(prompt)
    print("\nAI:", response)
    print("\nSources:")
    for s in sources:
        print("-", s)

print("\n--Multi document RAG Stopped--")