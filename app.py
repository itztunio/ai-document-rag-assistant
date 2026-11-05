import streamlit as st
import os
import numpy as np
import faiss
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="AI Document Q&A Assistant (RAG)",
    page_icon="📄",
    layout="wide"
)

# Load local environment variables if available
load_dotenv()
load_dotenv(".env.example")  # Backup: in case user edited .env.example directly

# Also search directly in project folder
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, ".env"))
load_dotenv(os.path.join(current_dir, ".env.example"))

def get_loaded_api_key():
    # 1. Check Streamlit Secrets (for cloud deployment)
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"].strip()
    except Exception:
        pass
        
    # 2. Check environment variables
    return (
        os.getenv("GEMINI_API_KEY") 
        or os.getenv("GOOGLE_API_KEY") 
        or ""
    ).strip()

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Helper Functions ----------------- #

def extract_text_from_pdf(pdf_docs):
    """Extracts raw text from multiple uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

def split_text_into_chunks(text, chunk_size=800, overlap=150):
    """Splits large text into smaller chunks with overlap for better context."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embeddings(texts, api_key):
    """Generates embeddings using Google Gemini's embedding model."""
    genai.configure(api_key=api_key)
    embeddings = []
    
    # Verified active embedding model
    for t in texts:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=t
        )
        embeddings.append(result['embedding'])
    return np.array(embeddings, dtype=np.float32)

def create_faiss_index(embeddings):
    """Creates a FAISS vector index from embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def query_vector_store(query, index, chunks, api_key, top_k=3):
    """Searches the FAISS index for the top-k most relevant chunks."""
    genai.configure(api_key=api_key)
    
    query_embedding = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query
    )['embedding']
    
    query_vector = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_vector, top_k)
    
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return relevant_chunks

import time

def generate_answer(query, relevant_chunks, api_key):
    """Generates an answer based strictly on retrieved context with automatic rate-limit handling."""
    genai.configure(api_key=api_key)
    
    # Models to try if one hits rate limits
    model_names = [
        "models/gemini-3.8-flash",
        "models/gemini-flash-latest",
        "models/gemini-3.1-flash-lite",
        "models/gemini-3.5-flash-lite"
    ]
    
    context = "\n\n---\n\n".join(relevant_chunks)
    prompt = f"""
You are an expert AI document analysis assistant. Answer the user's question accurately based ONLY on the provided context below.
If the answer cannot be found in the context, politely state: "The provided document does not contain sufficient information to answer this question."

Context:
{context}

Question:
{query}

Answer (clearly formatted with bullet points if applicable):
"""
    last_err = None
    for model_name in model_names:
        for attempt in range(3):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text
            except Exception as e:
                last_err = e
                err_msg = str(e).lower()
                # If rate-limited (429), wait and retry
                if "429" in err_msg or "quota" in err_msg:
                    time.sleep(5)
                    continue
                else:
                    break
                    
    raise Exception(f"Rate limit exceeded or service busy. Please wait 15 seconds and try again: {str(last_err)}")

# ----------------- UI Sidebar ----------------- #

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712038.png", width=70)
    st.title("Configuration")
    
    # API Key Configuration
    server_api_key = get_loaded_api_key()
    
    if server_api_key:
        api_key = server_api_key
        st.success("🔒 System API Connected")
        with st.expander("Use Custom API Key (Optional)"):
            custom_key = st.text_input(
                "Custom Gemini API Key",
                type="password",
                help="Optional: Override with your personal key."
            )
            if custom_key.strip():
                api_key = custom_key.strip()
    else:
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Get your free API key at https://aistudio.google.com/"
        )
    
    st.markdown("---")
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("🚀 Process Documents", use_container_width=True):
        if not api_key:
            st.error("Please enter your Gemini API Key first!")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Reading & indexing document(s)..."):
                try:
                    # 1. Extract Text
                    raw_text = extract_text_from_pdf(uploaded_files)
                    if not raw_text.strip():
                        st.error("Could not extract any text from the uploaded PDF(s).")
                    else:
                        # 2. Split into chunks
                        chunks = split_text_into_chunks(raw_text)
                        
                        # 3. Create Embeddings
                        embeddings = get_embeddings(chunks, api_key)
                        
                        # 4. Build Vector Index
                        index = create_faiss_index(embeddings)
                        
                        # Save in session state
                        st.session_state["faiss_index"] = index
                        st.session_state["chunks"] = chunks
                        st.session_state["doc_processed"] = True
                        st.session_state["messages"] = []
                        
                        st.success(f"Successfully processed! Created {len(chunks)} chunks.")
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

    st.markdown("---")
    st.caption("Built with Python, Streamlit, FAISS & Gemini API.")

# ----------------- Main Chat Interface ----------------- #

st.markdown('<div class="main-header">📄 Intelligent Document Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Retrieval-Augmented Generation (RAG) System — Upload documents and ask questions in natural language.</div>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User question input
user_question = st.chat_input("Ask a question about your uploaded document...")

if user_question:
    if not st.session_state.get("doc_processed", False):
        st.warning("Please upload and click 'Process Documents' in the sidebar first.")
    elif not api_key:
        st.error("Please provide your Google Gemini API Key.")
    else:
        # Add user query to chat history
        st.session_state["messages"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate response using RAG
        with st.chat_message("assistant"):
            with st.spinner("Searching document & formulating answer..."):
                try:
                    # Retrieve chunks
                    relevant_chunks = query_vector_store(
                        user_question,
                        st.session_state["faiss_index"],
                        st.session_state["chunks"],
                        api_key
                    )
                    
                    # Generate response
                    answer = generate_answer(user_question, relevant_chunks, api_key)
                    
                    st.markdown(answer)
                    
                    # Collapsible section to view retrieved source passages
                    with st.expander("🔍 View Retrieved Context / Sources"):
                        for idx, chunk in enumerate(relevant_chunks, 1):
                            st.markdown(f"**Chunk {idx}:**")
                            st.info(chunk)
                    
                    # Add assistant response to history
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Failed to generate answer: {str(e)}")
