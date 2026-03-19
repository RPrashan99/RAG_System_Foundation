import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Load embeddings and FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

st.set_page_config(
    page_title="Multi PDF RAG Assistant",
    layout="wide"
)

st.title("Multi PDF RAG Assistant")

try:
    llm = OllamaLLM(model="llama3")
except:
    st.write('## Model not loaded. Check Ollama is running.')

col1, col2 = st.columns([3, 2])

with col1:
    query = st.text_input("Ask a question:")

    bar = st.progress(0, text="Working...")

    if query:
        docs = db.similarity_search(query, k=6)

        bar.progress(30, text=f"Working... 30%")

        context = ""
        sources = set()

        for d in docs:
            context += d.page_content + "\n\n"
            sources.add(d.metadata["source"])

        bar.progress(50, text=f"Working... 50%")

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
        bar.progress(100, text=f"Working... 100%")
        st.write("### Answer")
        st.write(response)
        st.write('### Source')
        for s in sources:
            st.write(s)
        bar.empty()

with col2:
    st.write("### Instructions")
    st.write("""
    - Enter a question related to the PDF content.
    - The assistant will use the most relevant sections of the document to answer.
    - Ensure your question is clear and specific for better results.
    """)
