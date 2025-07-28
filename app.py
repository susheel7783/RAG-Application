import streamlit as st
import time
import os

# Set your Google API key here (replace with your actual key)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDG740XOyQw8fGt_4SULs6ue6c6g8MteRg"

# Simple PDF text extraction and chunking - NO LANGCHAIN DEPENDENCIES
def extract_and_chunk_pdf(pdf_path, chunk_size=1000, overlap=200):
    """Extract text from PDF and split into chunks"""
    try:
        import pypdf
        
        # Read PDF
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            full_text = ""
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():
                    full_text += text + "\n"
        
        # Simple text chunking
        chunks = []
        words = full_text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append({
                    'text': chunk,
                    'metadata': {'chunk_id': len(chunks), 'source': pdf_path}
                })
        
        return chunks
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return []

# Simple vector search using sentence transformers
def create_embeddings(texts):
    """Create embeddings for texts"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([chunk['text'] for chunk in texts])
        return embeddings, model
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None, None

def find_relevant_chunks(query, chunks, embeddings_model, embeddings, top_k=5):
    """Find relevant chunks for a query"""
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get query embedding
        query_embedding = embeddings_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = [chunks[i]['text'] for i in top_indices]
        return relevant_chunks
    except Exception as e:
        st.error(f"Error finding relevant chunks: {str(e)}")
        return []

from langchain_google_genai import ChatGoogleGenerativeAI

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

# Cache the system initialization
@st.cache_resource
def initialize_system():
    """Initialize the RAG system"""
    try:
        # Check if PDF file exists
        pdf_file = "yolov9_paper.pdf"
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"PDF file '{pdf_file}' not found. Please ensure it's in the same directory as app.py")
        
        # Extract and chunk PDF
        chunks = extract_and_chunk_pdf(pdf_file)
        
        if not chunks:
            raise ValueError("No chunks created from PDF file")
        
        # Create embeddings
        embeddings, embeddings_model = create_embeddings(chunks)
        
        if embeddings is None:
            raise ValueError("Failed to create embeddings")
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=30
        )
        
        return chunks, embeddings, embeddings_model, llm, len(chunks)
        
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        raise e

# Initialize components
if not st.session_state.system_initialized:
    with st.spinner("🔄 Initializing RAG system... This may take a moment."):
        try:
            chunks, embeddings, embeddings_model, llm, chunk_count = initialize_system()
            
            # Store in session state
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.embeddings_model = embeddings_model
            st.session_state.llm = llm
            st.session_state.chunk_count = chunk_count
            st.session_state.system_initialized = True
            
            st.success(f"✅ RAG system initialized successfully! Processed {chunk_count} document chunks.")
            
        except Exception as e:
            st.error(f"❌ Error initializing RAG system: {str(e)}")
            st.error("Please check that:")
            st.error("1. Your GOOGLE_API_KEY is set correctly in the code")
            st.error("2. The yolov9_paper.pdf file exists in the app directory")
            st.error("3. All required packages are installed")
            st.stop()

# Get components from session state
chunks = st.session_state.chunks
embeddings = st.session_state.embeddings
embeddings_model = st.session_state.embeddings_model
llm = st.session_state.llm

# Sidebar with information
with st.sidebar:
    st.header("📋 System Info")
    if st.session_state.system_initialized:
        st.success("System Status: ✅ Ready")
        st.info(f"Documents processed: {st.session_state.chunk_count}")
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

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat interface
if query := st.chat_input("Ask your question about the document..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Processing your query..."):
            try:
                start_time = time.time()
                
                # Find relevant chunks
                relevant_chunks = find_relevant_chunks(
                    query, chunks, embeddings_model, embeddings, top_k=5
                )
                
                if not relevant_chunks:
                    answer = "I couldn't find relevant information to answer your question."
                else:
                    # Create context from relevant chunks
                    context = "\n\n".join(relevant_chunks)
                    
                    # Create prompt
                    prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.

Context:
{context}

Question: {query}

Answer:"""
                    
                    # Get response from LLM
                    response = llm.invoke(prompt)
                    answer = response.content if hasattr(response, 'content') else str(response)
                
                end_time = time.time()
                
                # Display the answer
                st.markdown(answer)
                
                # Show processing time
                processing_time = round(end_time - start_time, 2)
                st.caption(f"⏱️ Response generated in {processing_time} seconds")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer
                })
                
                # Optional: Show retrieved context
                if relevant_chunks:
                    with st.expander("📄 View Retrieved Context", expanded=False):
                        for i, chunk in enumerate(relevant_chunks):
                            st.write(f"**Chunk {i+1}:**")
                            st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)
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
