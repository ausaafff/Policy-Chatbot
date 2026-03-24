import os
import shutil
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st
from PyPDF2 import PdfReader
from google import genai
from google.genai import types

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# Environment & Client Setup
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="Policy Chatbot",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

# Constants
MODEL_ID = "gemini-3-flash-preview"
EMBED_MODEL = "gemini-embedding-001"
PDF_FOLDER = "./pdfs"
VECTORSTORE_PATH = "./faiss_index"

os.makedirs(PDF_FOLDER, exist_ok=True)

# CSS Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2.5rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f8f9fa;
    border-right: 1px solid #e0e0e0;
}
[data-testid="stSidebar"] * { color: #000 !important; }
[data-testid="stSidebar"] .stButton button {
    margin: 10px 0 !important;
    background: #007bff !important;
    border: none !important;
    color: #fff !important;
    border-radius: 5px !important;
    font-size: 14px !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease;
}
[data-testid="stSidebar"] .stButton button:hover {
    background-color: #0056b3 !important;
}

/* Main */
.page-header { margin-bottom: 2rem; }
.page-title  { font-size: 28px; color: #333; margin: 0; }
.page-subtitle { font-size: 15px; color: #666; margin-top: 5px; }

.chat-box { border: 1px solid #e0e0e0; margin-top: 20px; padding: 15px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Embeddings
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            results.extend(e.values for e in response.embeddings)
        return results

    def embed_query(self, text: str) -> list[float]:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return response.embeddings[0].values

# PDF Helpers
def load_pdfs():
    documents = []
    pdf_files = sorted(f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf"))
    if not pdf_files:
        return []
    progress = st.progress(0)
    for i, filename in enumerate(pdf_files):
        progress.progress((i + 1) / len(pdf_files))
        reader = PdfReader(os.path.join(PDF_FOLDER, filename))
        for j, page in enumerate(reader.pages):
            text = page.extract_text().strip()
            if text:
                documents.append(Document(page_content=text, metadata={"source": filename, "page": j + 1}))
    return documents

def split_and_vectorize_docs(documents):
    # Import updated module (fix for Render issue)
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # Create vectorstore
    vectorstore = FAISS.from_documents(
        chunks,
        embedding=GeminiEmbeddings()
    )

    # Save safely
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    return vectorstore

def load_vectorstore():
    if not os.path.exists(VECTORSTORE_PATH):
        return None
    return FAISS.load_local(
        VECTORSTORE_PATH,
        GeminiEmbeddings(),
        allow_dangerous_deserialization=True
    )
# Answer Generation
def gemini_answer(context, question):
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=f"Context:\n{context}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a precise document analyst. Answer using ONLY the provided context.\n\n"
                    "RULES:\n"
                    "1. Be concise and direct — no filler phrases.\n"
                    "2. Use bullet points for multi-part answers.\n"
                    "3. Cite sources inline as [filename, p.X].\n"
                    "4. If not found, respond with: 'Not found in the uploaded documents.'"
                ),
            ),
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

def run_qa(vectorstore, question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(f"[{d.metadata['source']} | Page {d.metadata['page']}]\n{d.page_content}" for d in docs)
    answer = gemini_answer(context, question)
    return {"answer": answer, "sources": docs}

# Session State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.title("Policy Chatbot 📑")
    st.write("Askquestions about policies and get answers")

    if st.button("Index PDFs"):
        docs = load_pdfs()
        if docs:
            st.session_state.vectorstore = split_and_vectorize_docs(docs)
            st.success("PDFs indexed successfully!")
        else:
            st.error("No PDFs found to index.")
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")
    
    if st.button("Reset Index"):
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)
        st.session_state.vectorstore = None
        st.success("Index reset successfully!")

# Main Content
st.markdown("""
<div class='page-header'>
    <h1 class='page-title'>Ask Your Documents</h1>
    <p class='page-subtitle'>Receive answers grounded strictly in your PDFs</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.vectorstore:
    st.info("No documents indexed yet. Please index your PDFs first.")
    st.stop()

for chat in st.session_state.chat_history:
    st.write(f"**Q:** {chat['question']}")
    st.write(f"**A:** {chat['answer']}", unsafe_allow_html=True)

question = st.text_input("Ask anything about your documents...")

if question:
    with st.spinner("Generating response..."):
        result = run_qa(st.session_state.vectorstore, question)
    st.session_state.chat_history.append({"question": question, "answer": result["answer"]})
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write(f"**Q:** {question}")
    st.write(f"**A:** {result['answer']}", unsafe_allow_html=True)