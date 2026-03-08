from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Load embeddings
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = OllamaLLM(model="llama3")

while True:
    query = input("\nAsk: ")

    docs = db.similarity_search(query, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are answering questions using document context.

Rules:
- Use ONLY the provided context.
- Tables may appear as text.
- Preserve numeric relationships carefully.
- Do NOT guess missing values.

Context:
{context}

Question: {query}
"""

    response = llm.invoke(prompt)

    print("\nAI:", response)


