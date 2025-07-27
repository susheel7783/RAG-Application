
import streamlit as st
import asyncio
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Fix for asyncio event loop issue
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

st.title("RAG Application built on Gemini Model")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache the vectorstore creation to avoid recreating it on every run
@st.cache_resource
def create_vectorstore():
    """Create and cache the vectorstore"""
    loader = PyPDFLoader("yolov9_paper.pdf")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Use sync version of embeddings initialization
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document"
    )
    
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings
    )
    
    return vectorstore

# Cache the LLM initialization
@st.cache_resource
def create_llm():
    """Create and cache the LLM"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None
    )

# Initialize components
try:
    vectorstore = create_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    llm = create_llm()
    
except Exception as e:
    st.error(f"❌ Error initializing RAG system: {str(e)}")
    st.stop()

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

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat interface
if query := st.chat_input("Ask Your Query:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            try:
                # Create the chains
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                # Get response
                response = rag_chain.invoke({"input": query})
                
                # Display the answer
                answer = response["answer"]
                st.markdown(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer
                })
                        
            except Exception as e:
                error_message = f"❌ Error processing query: {str(e)}"
                st.error(error_message)
                # Add error to chat history
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Add a button to clear chat history
if st.button("🔄 Clear Chat History"):
    st.session_state.messages = []
    st.rerun()



# new interface and features added in the below 
# import streamlit as st
# import asyncio
# import time
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# # Fix for asyncio event loop issue
# try:
#     loop = asyncio.get_event_loop()
# except RuntimeError:
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)

# st.title("RAG Application built on Gemini Model")

# # Cache the vectorstore creation to avoid recreating it on every run
# @st.cache_resource
# def create_vectorstore():
#     """Create and cache the vectorstore"""
#     loader = PyPDFLoader("yolov9_paper.pdf")
#     data = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
#     docs = text_splitter.split_documents(data)

#     # Use sync version of embeddings initialization
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001",
#         task_type="retrieval_document"
#     )
    
#     vectorstore = Chroma.from_documents(
#         documents=docs, 
#         embedding=embeddings
#     )
    
#     return vectorstore

# # Cache the LLM initialization
# @st.cache_resource
# def create_llm():
#     """Create and cache the LLM"""
#     return ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=0,
#         max_tokens=None,
#         timeout=None
#     )

# # Initialize components
# try:
#     vectorstore = create_vectorstore()
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
#     llm = create_llm()
    
#     st.success("✅ RAG system initialized successfully!")
    
# except Exception as e:
#     st.error(f"❌ Error initializing RAG system: {str(e)}")
#     st.stop()

# # Create the prompt template
# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )

# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# # Chat interface
# query = st.chat_input("Ask Your Query:")

# if query:
#     with st.spinner("Processing your query..."):
#         try:
#             # Create the chains
#             question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
#             rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#             # Get response
#             response = rag_chain.invoke({"input": query})
            
#             # Display the answer
#             st.write("**Answer:**")
#             st.write(response["answer"])
            
#             # Optionally show retrieved context (for debugging)
#             if st.checkbox("Show retrieved context"):
#                 st.write("**Retrieved Context:**")
#                 for i, doc in enumerate(response["context"]):
#                     st.write(f"**Document {i+1}:**")
#                     st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
#                     st.write("---")
                    
#         except Exception as e:
#             st.error(f"❌ Error processing query: {str(e)}")

# # Add some helpful information
# # with st.sidebar:
# #     st.header("ℹ️ About")
# #     st.write("This RAG application uses Google's Gemini model to answer questions .")
# #     st.write("**Features:**")
# #     st.write("- PDF document processing")
# #     st.write("- Vector similarity search")
# #     st.write("- Conversational AI responses")
    
# #     if st.button("Clear Cache"):
# #         st.cache_resource.clear()
# #         st.success("Cache cleared! Please refresh the page.")