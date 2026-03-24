# 📑 Policy Chatbot – Intelligent Poilicy Question Answering System

Policy Chatbot is a production-ready document intelligence application that enables users to interact with PDF-based knowledge sources through natural language queries. It leverages semantic search and large language models to deliver precise, context-grounded answers with source attribution.

---

## 🚀 Overview

Policy Chatbottransforms static documents into an interactive knowledge system by combining vector search with generative AI. It ensures that all responses are strictly derived from the uploaded documents, making it suitable for policy analysis, compliance systems, and enterprise knowledge retrieval.

---

## ✨ Key Features

- Semantic Document Search using vector embeddings
- Context-Aware Responses powered by generative AI
- Source Attribution with filename and page references
- Multi-PDF Support for scalable knowledge ingestion
- Interactive Chat Interface with session history
- Index Management (re-index, reset, clear chat)
- Strict Context Grounding (no hallucinated answers)

---

## 🧠 System Architecture

```
User Query
    ↓
Retriever (FAISS)
    ↓
Relevant Document Chunks
    ↓
LLM (Gemini)
    ↓
Final Answer with Citations
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| Document Parsing | PyPDF2 |
| Text Chunking | LangChain |
| Vector Database | FAISS |
| Embeddings & LLM | Google Gemini API |

---

## 📁 Project Structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Dependencies
├── .env                # API key (never commit)
├── pdfs/               # Source PDF documents
├── faiss_index/        # Stored vector index
└── README.md           # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone <repository_url>
cd <project_directory>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_api_key_here
```
---

## ▶️ Running Locally

```bash
streamlit run app.py
```

---

## ☁️ Deployment on Render

**Build Command**
```bash
pip install -r requirements.txt
```

**Start Command**
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

**Environment Variable** (set in Render Dashboard)
```
GOOGLE_API_KEY=your_api_key_here
```

---

## 📌 Usage Workflow

1. Place PDF documents inside the `pdfs/` directory
2. Launch the application with `streamlit run app.py`
3. Click **Index PDFs** in the sidebar to build the vector store
4. Enter queries in natural language
5. Receive precise, context-bound answers with citations

---

## ⚠️ Important Considerations

- Sensitive data should not be committed to version control
- `.env` files must remain local and never pushed to GitHub
- Vector index persistence depends on hosting configuration
- Free-tier deployments may experience cold-start latency

---

## 📈 Future Enhancements

- Dynamic PDF upload via UI
- Role-based access control
- Incremental indexing for large datasets
- Advanced retrieval strategies (hybrid search)
- Performance optimization for large-scale deployments

---

## 🧑‍💻 Author

Developed as a scalable document intelligence solution leveraging modern NLP and retrieval-based architectures.