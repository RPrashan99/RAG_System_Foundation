import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

st.set_page_config(
    page_title="Conversational RAG Assistant",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Conversational PDF RAG Assistant")

try:
    llm = OllamaLLM(model="llama3")
except Exception as e:
    st.error("Model not loaded. Ensure Ollama is running.")
    llm = None

col1, col2 = st.columns([3, 2])

with col1:
    query = st.text_input("Ask a question:")

    if query and llm:
        with st.spinner("Thinking..."):

            docs = db.similarity_search(query, k=6)

            context = ""
            sources = set()
            for d in docs:
                context += d.page_content + "\n\n"
                sources.add(d.metadata.get("source", "Unknown"))

            history_text = "\n".join(
                [f"User: {q}\nAssistant: {a}" for q, a in st.session_state.chat_history]
            )

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

        st.write("### Answer")
        st.write(response)

        st.write("### Sources")
        for s in sources:
            st.markdown(f"- [{s}]({s})")

        st.session_state.chat_history.append((query, response))

with col2:
    st.write("### Instructions")
    st.write("""
    - Enter a question related to the PDF content.
    - The assistant will use the most relevant sections of the document to answer.
    - Ensure your question is clear and specific for better results.
    """)

with st.sidebar:
    st.write("### Chat History")
    if st.session_state.chat_history:
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            preview = a[:100] + "..." if len(a) > 100 else a
            with st.expander(f"Assistant: {preview}"):
                st.write(a)
    else:
        st.write("No conversation yet.")
