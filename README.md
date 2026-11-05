# 📄 Intelligent Document Q&A Assistant (RAG System)

An end-to-end **Retrieval-Augmented Generation (RAG)** application built with Python, Streamlit, FAISS, and Google Gemini. This tool enables users to upload PDF documents and ask questions in natural language, retrieving accurate, context-grounded answers with source citations.

---

## 🌟 Key Features
- **Multi-Document PDF Ingestion:** Upload one or multiple PDFs seamlessly.
- **Smart Chunking & Overlap:** Preserves semantic context across document sections.
- **Vector Search with FAISS:** Uses Google Gemini `text-embedding-004` to create vector embeddings and perform lightning-fast similarity search.
- **Context-Grounded LLM Responses:** Powered by `gemini-1.5-flash` with strict prompting to eliminate hallucinations.
- **Source Inspection:** Expandable views showing the exact text chunks used to answer each query.
- **Clean Interactive UI:** Built with Streamlit chat components.

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Frontend / Framework:** Streamlit
- **LLM & Embeddings:** Google Generative AI (Gemini 1.5 Flash & Text Embedding 004)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Document Parser:** PyPDF

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/itztunio/ai-document-rag-assistant.git
cd ai-document-rag-assistant
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Create a `.env` file in the root directory:
```env
GEMINI_API_KEY="your_api_key_here"
```
*(Alternatively, you can enter the API key directly in the application sidebar).*

### 5. Run the Application
```bash
streamlit run app.py
```

---

## 📸 Usage Walkthrough
1. **Launch:** Run `streamlit run app.py` and open `http://localhost:8501`.
2. **Configure:** Enter your Gemini API key in the sidebar.
3. **Upload:** Upload your PDF documents and click **"Process Documents"**.
4. **Chat:** Ask any question in the chat box to get instant, referenced answers!

---

## 👤 Author
- **Ayesha Tunio**
- GitHub: [@itztunio](https://github.com/itztunio)
- LinkedIn: [Ayesha Tunio](https://linkedin.com/in/ayesha-tunio-6b2a88382)
