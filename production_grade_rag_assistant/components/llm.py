from langchain_ollama import OllamaLLM

def load_llm(model_name="llama3"):
    try:
        llm = OllamaLLM(model=model_name)
        return llm
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
def generate_response(llm, prompt):
    if llm is None:
        return "Model not available."
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error generating response."
    
def generate_prompt(query, context, history):
    history_text = "\n".join(
        [f"User: {q}\nAssistant: {a}" for q, a in history]
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
    return prompt

def rewrite_query(llm, query, history):
    
    history_text = "\n".join(
        [f"User: {q}\nAssistant: {a}" for q, a in history]
    )

    rewrite_prompt = f"""
        Rewrite the user query so it is self-contained.

        conversation history:
        {history_text}

        User Question:
        {query}

        Rewritten Query:
    """
    rewritten_query = generate_response(llm, rewrite_prompt)
    return rewritten_query.strip()

def rewrite_and_generate_response(llm, query, context, history):
    rewritten_query = rewrite_query(llm, query, history)
    prompt = generate_prompt(rewritten_query, context, history)
    response = generate_response(llm, prompt)
    return response