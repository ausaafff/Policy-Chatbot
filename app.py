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
    page_title="SNEHA Policy Q&A",
    page_icon="📑",
    layout="centered", 
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

# CSS Styling (Minimal design)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 800px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #fafafa;
    border-right: 1px solid #eaeaea;
}
[data-testid="stSidebar"] .stButton button {
    width: 100%;
    margin: 5px 0 !important;
    background: #ffffff !important;
    border: 1px solid #dcdcdc !important;
    color: #333 !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease;
}
[data-testid="stSidebar"] .stButton button:hover {
    border-color: #999 !important;
    background-color: #f0f0f0 !important;
}

/* Header & Minimal UI */
.header-container { text-align: center; margin-bottom: 2.5rem; }
.page-title  { font-size: 24px; font-weight: 600; color: #111; margin: 0; }
.page-subtitle { font-size: 14px; color: #666; margin-top: 4px; font-weight: 300; }

/* Chat Elements */
.user-msg { background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #eee; }
.bot-msg { background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #e5e5e5; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
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
    docs = retriever.invoke(question)
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
    # SNEHA Logo moved here
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <img src="https://iili.io/frR6CJa.md.png" style="max-width: 140px;" alt="SNEHA Logo">
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Admin Panel")
    
    if st.button("Index Policy PDFs"):
        docs = load_pdfs()
        if docs:
            st.session_state.vectorstore = split_and_vectorize_docs(docs)
            st.success("PDFs indexed successfully!")
        else:
            st.error("No PDFs found to index.")
    
    if st.button("Clear Current Chat"):
        st.session_state.chat_history = []
        st.rerun() 
    
    if st.button("Reset Index"):
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)
        st.session_state.vectorstore = None
        st.success("Index reset successfully!")

# Main Content (Clean Header)
st.markdown("""
<div class='header-container'>
    <h1 class='page-title'>Policy Q&A</h1>
    <p class='page-subtitle'>Ask questions and get precise answers from company policies.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.vectorstore:
    st.info("No policy documents indexed yet. Please index your PDFs using the sidebar.")
    st.stop()

# Input box at the top of the interaction area
question = st.text_input("What would you like to know?", placeholder="e.g., What is the leave policy?", label_visibility="collapsed")

if question:
    with st.spinner("Analyzing SNEHA policies..."):
        result = run_qa(st.session_state.vectorstore, question)
    
    # We still append to history logic-wise, but we will ONLY display the latest one
    st.session_state.chat_history.append({"question": question, "answer": result["answer"]})

# Display ONLY the latest chat interaction
if st.session_state.chat_history:
    latest_chat = st.session_state.chat_history[-1]
    
    st.markdown(f"""
    <div class="user-msg">
        <strong>You:</strong><br>{latest_chat['question']}
    </div>
    <div class="bot-msg">
        <strong>SNEHA Bot:</strong><br>{latest_chat['answer']}
    </div>
    """, unsafe_allow_html=True)