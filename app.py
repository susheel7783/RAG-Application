import streamlit as st
import asyncio
import time
import os

# Set your Google API key here (replace with your actual key)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDG740XOyQw8fGt_4SULs6ue6c6g8MteRg"

# Simple manual PDF text extraction - no dependencies needed
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pypdf (which you already have installed)"""
    try:
        import pypdf
        documents = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    # Create a simple document-like object
                    doc_content = {
                        'page_content': text,
                        'metadata': {'page': page_num + 1, 'source': pdf_path}
                    }
                    documents.append(doc_content)
        
        return documents
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return []

# Convert our simple docs to langchain Document format
def create_langchain_documents(simple_docs):
    """Convert simple docs to langchain Document format"""
    from langchain.schema import Document
    
    langchain_docs = []
    for doc in simple_docs:
        langchain_doc = Document(
            page_content=doc['page_content'],
            metadata=doc['metadata']
        )
        langchain_docs.append(langchain_doc)
    
    return langchain_docs

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import Chroma
try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain.vectorstores import Chroma
    except ImportError:
        # Fallback to basic implementation
        st.error("Chroma not available. Please install: pip install chromadb")
        st.stop()

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
    st.info("Get your API key from: https://makersuite.google.com/app/apikey")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize system state
if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False

# Cache the vectorstore creation to avoid recreating it on every run
@st.cache_resource
def create_vectorstore():
    """Create and cache the vectorstore"""
    try:
        # Check if PDF file exists
        pdf_file = "yolov9_paper.pdf"
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"PDF file '{pdf_file}' not found. Please ensure it's in the same directory as app.py")
        
        # Load PDF using our simple function
        simple_docs = extract_text_from_pdf(pdf_file)
        
        if not simple_docs:
            raise ValueError("No data loaded from PDF file")
        
        # Convert to langchain format
        data = create_langchain_documents(simple_docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)
        
        if not docs:
            raise ValueError("No documents created after splitting")

        # Use embeddings
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
            st.error(f"❌ Error initializing RAG system: {str(e)}")
            st.error("Please check that:")
            st.error("1. Your GOOGLE_API_KEY is set correctly in the code")
            st.error("2. The yolov9_paper.pdf file exists in the app directory")
            st.error("3. All required packages are installed")
            st.stop()

# Get components from session state
retriever = st.session_state.retriever
llm = st.session_state.llm

# Create the prompt template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Sidebar with information
with st.sidebar:
    st.header("📋 System Info")
    if st.session_state.system_initialized:
        st.success("System Status: ✅ Ready")
        st.info(f"Documents processed: {st.session_state.doc_count}")
    else:
        st.warning("System Status: ⏳ Initializing")
    
    st.header("ℹ️ About")
    st.write("This RAG application uses Google's Gemini model to answer questions about the uploaded PDF document.")
    
    st.write("**Features:**")
    st.write("- PDF document processing")
    st.write("- Vector similarity search")
    st.write("- Conversational AI responses")
    st.write("- Chat history")
    
    if st.button("🗑️ Clear All Cache"):
        st.cache_resource.clear()
        st.session_state.clear()
        st.success("Cache cleared! Please refresh the page.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat interface
if query := st.chat_input("Ask your question about the document..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("🔍 Processing your query..."):
            try:
                # Create the chains
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                # Get response with timeout
                start_time = time.time()
                response = rag_chain.invoke({"input": query})
                end_time = time.time()
                
                # Display the answer
                answer = response["answer"]
                st.markdown(answer)
                
                # Show processing time
                processing_time = round(end_time - start_time, 2)
                st.caption(f"⏱️ Response generated in {processing_time} seconds")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer
                })
                
                # Optional: Show retrieved context in an expander
                with st.expander("📄 View Retrieved Context", expanded=False):
                    for i, doc in enumerate(response.get("context", [])):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.caption(f"Source: {doc.metadata}")
                        st.divider()
                        
            except Exception as e:
                error_message = f"❌ Error processing query: {str(e)}"
                st.error(error_message)
                st.error("This might be due to:")
                st.error("- API rate limits")
                st.error("- Network connectivity issues")
                st.error("- Invalid query format")
                
                # Add error to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_message
                })

# Footer with controls
st.divider()
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("🔄 Restart System"):
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()
