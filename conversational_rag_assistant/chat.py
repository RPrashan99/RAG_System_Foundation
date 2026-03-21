from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("\n---Conversational RAG Assistant---")

try:
    llm = OllamaLLM(model="llama3")
except:
    print("\nModel not loaded. Check Ollama is running.")

# save previous chats
chat_history = []

while True:
    query = input("\nAsk (Type 'q' to quit): ")

    if(query.lower() == 'q'):
        break

    docs = db.similarity_search(query, k=6)

    context = ""
    sources = set()

    for d in docs:
        context += d.page_content + "\n\n"
        sources.add(d.metadata["source"])

    # create previous chats from chat history
    history_text = []

    for q, a in chat_history:
        history_text += f'User: {q}\nAssistant: {a}\n'

    prompt = f"""
        You are a helpful AI assistant.

        Use the conversation history and context to answer.

        Conversation History:
        {history_text}

        Context from documents:
        {context}

        User Question:
        {query}

        Answer briefly and clearly using the context.
        """
    
    response = llm.invoke(prompt)
    print('\nAI: ', response)
    print("\nSources:")
    for s in sources:
        print("-", s)

    # Add new question and answer to history
    chat_history.append((query, response))

print("\n---Conversational RAG Assistant Stopped---")