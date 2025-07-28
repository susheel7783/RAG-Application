import streamlit as st
import asyncio
import time
import os

# Set your Google API key here (replace with your actual key)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDG740XOyQw8fGt_4SULs6ue6c6g8MteRg"

# Alternative PDF loader that works without langchain-community
import PyPDF2
from io import BytesIO
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import Chroma with fallback
try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain.vectorstores import Chroma
    except ImportError:
        from langchain.vectorstores.chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Fix for asyncio event loop issue
try:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        raise RuntimeError("Event loop is closed")
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Page configuration
st.set_page_config(
    page_title="RAG Application - Gemini Model",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Application built on Gemini Model")

# Check for required environment variables
if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_actual_google_api_key_here":
    st.error("❌ Please set your actual GOOGLE_API_KEY in the code on line 7")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize system state
if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False

# Custom PDF loader function (replaces PyPDFLoader)
def load_pdf_with_pypdf2(file_path):
    """Load PDF using PyPDF2 and return Document objects"""
    documents = []
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    doc = Document(
                        page_content=text,
                        metadata={"page": page_num + 1, "source": file_path}
                    )
                    documents.append(doc)
    
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return []
    
    return documents

# Cache the vectorstore creation to avoid recreating it on every run
@st.cache_resource
def create_vectorstore():
    """Create and cache the vectorstore"""
    try:
        # Check if PDF file exists
        pdf_file = "yolov9_paper.pdf"
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"PDF file '{pdf_file}' not found. Please ensure it's in the same directory as app.py")
        
        # Load PDF using our custom function
        data = load_pdf_with_pypdf2(pdf_file)
        
        if not data:
            raise ValueError("No data loaded from PDF file")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)
        
        if not docs:
            raise ValueError("No documents created after splitting")

        # Use sync version of embeddings initialization
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document"
        )
        
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings
        )
        
        return vectorstore, len(docs)
        
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        raise e

# Cache the LLM initialization
@st.cache_resource
def create_llm():
    """Create and cache the LLM"""
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=30
        )
    except Exception as e:
        st.error(f"Error creating LLM: {str(e)}")
        raise e

# Initialize components with better error handling
if not st.session_state.system_initialized:
    with st.spinner("🔄 Initializing RAG system... This may take a moment."):
        try:
            vectorstore, doc_count = create_vectorstore()
            retriever = vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 5}
            )
            llm = create_llm()
            
            # Store in session state for reuse
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.llm = llm
            st.session_state.doc_count = doc_count
            st.session_state.system_initialized = True
            
            st.success(f"✅ RAG system initialized successfully! Processed {doc_count} document chunks.")
            
        except Exception as e:
            st.error(
