import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Load embeddings and FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

llm = OllamaLLM(model="llama3")

st.set_page_config(
    page_title="Simple RAG PDF Assistant",
    layout="wide"
)

st.title("Simple RAG PDF Assistant")

col1, col2 = st.columns([3, 2])

with col1:
    query = st.text_input("Ask a question:")

    bar = st.progress(0, text="Working...")

    if query:
        docs = db.similarity_search(query, k=3)
        bar.progress(33, text=f"Working... 33%")

        context = "\n\n".join([d.page_content for d in docs])
        bar.progress(66, text=f"Working... 66%")

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
        bar.progress(100, text=f"Working... 100%")
        st.write("### Answer")
        st.write(response)
        bar.empty()

with col2:
    st.write("### Instructions")
    st.write("""
    - Enter a question related to the PDF content.
    - The assistant will use the most relevant sections of the document to answer.
    - Ensure your question is clear and specific for better results.
    """)
