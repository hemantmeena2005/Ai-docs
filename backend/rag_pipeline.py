# rag_pipeline.py

import os

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq



chat_history = []


llm = ChatGroq(
    model_name="llama-3.1-8b-instant"
)
# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

VECTOR_PATH = "vectorstore"


def create_vector_store(file_path: str):
    """
    Load PDF → split → create embeddings → store in FAISS
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_PATH)

    return "Vector store created successfully"


def load_vector_store():
    """
    Load existing FAISS index
    """
    if not os.path.exists(VECTOR_PATH):
        return None
    return FAISS.load_local(
    VECTOR_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

def ask_question(query: str):
    try:
        db = load_vector_store()

        if db is None:
            return "No document uploaded yet."

        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Build conversation history string
        history_text = ""
        for q, a in chat_history[-3:]:  # last 3 messages only
            history_text += f"User: {q}\nAssistant: {a}\n"

        prompt = f"""
        You are an AI study assistant.

        Conversation history:
        {history_text}

        Context:
        {context}

        Current question:
        {query}

        Answer clearly and concisely.
        """

        response = llm.invoke(prompt)

        answer = response.content

        # Save to memory
        chat_history.append((query, answer))

        return answer

    except Exception as e:
        return f"ERROR: {str(e)}"

def generate_summary():
    try:
        db = load_vector_store()

        if db is None:
            return "No document uploaded yet."

        docs = db.similarity_search("", k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are an AI study assistant.

        Summarize the following content in:
        - short paragraph
        - key bullet points

        Content:
        {context}
        """

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:
        return f"ERROR: {str(e)}"
    

def generate_quiz():
    try:
        db = load_vector_store()

        if db is None:
            return "No document uploaded yet."

        docs = db.similarity_search("", k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Generate 5 multiple-choice questions from the content.

        Format:
        Q1.
        A)
        B)
        C)
        D)
        Answer:

        Content:
        {context}
        """

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:
        return f"ERROR: {str(e)}"