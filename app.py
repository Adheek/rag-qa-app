import streamlit as st
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# Import all processing functions from config
from config import (
    config,
    get_embeddings_model,
    process_pdf_files,
    process_youtube_url,
    create_vector_store
)

load_dotenv()

# ─── Page Configuration ───
st.set_page_config(page_title="Academic QA", layout="centered")
st.title("🎓 Academic QA - RAG System")

# ─── Initialize Session State ───
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_chat_session"

# ─── Helper Functions ───

@st.cache_resource
def get_chat_model():
    """Initialize and cache the ChatGroq model."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=400
    )

def call_model(state: MessagesState):
    """LangGraph node function to call the LLM."""
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    model = get_chat_model()
    response = model.invoke(messages)
    return {"messages": response}

@st.cache_resource
def get_langgraph_app():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# ─── STEP 1: Upload Section ───
st.header("📤 Step 1: Upload Your Documents")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more PDF files to create your knowledge base"
)

youtube_url = st.text_input(
    "YouTube Video URL (optional)",
    placeholder="https://www.youtube.com/watch?v=...",
    help="Enter a YouTube URL to include video transcript"
)

# ─── STEP 2: Process Button ───
if st.button("🔄 Process Documents", type="primary", disabled=not (uploaded_files or youtube_url)):
    with st.spinner("Processing documents..."):
        try:
            all_documents = []

            # Process PDFs using config function
            if uploaded_files:
                st.info(f"Processing {len(uploaded_files)} PDF file(s)...")
                pdf_docs = process_pdf_files(uploaded_files)
                all_documents.extend(pdf_docs)
                st.success(f"✅ Loaded {len(pdf_docs)} pages from PDFs")

            # Process YouTube using config function
            if youtube_url:
                st.info("Processing YouTube video...")
                yt_docs = process_youtube_url(youtube_url)
                all_documents.extend(yt_docs)
                st.success(f"✅ Loaded YouTube transcript")

            # Create vector store using config function
            if all_documents:
                st.info("Creating vector database...")
                embeddings = get_embeddings_model()
                st.session_state.vectordb = create_vector_store(all_documents, embeddings)
                st.success(f"✅ Vector database created!")
                st.session_state.messages = []  # Reset chat
                st.rerun()
            else:
                st.error("No documents to process!")

        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

# ─── STEP 3: Chat Interface ───
if st.session_state.vectordb is not None:
    st.header("💬 Step 2: Ask Questions")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant documents
        with st.spinner("Searching..."):
            docs = st.session_state.vectordb.similarity_search(prompt, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])

        # Generate response
        full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"
        input_message = {"messages": [HumanMessage(content=full_prompt)]}
        config_dict = {"configurable": {"thread_id": st.session_state.thread_id}}

        with st.spinner("Generating answer..."):
            # Get cached app instance (maintains memory across questions)
            app = get_langgraph_app()
            response = app.invoke(input_message, config_dict)
            assistant_message = response["messages"][-1].content

        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        with st.chat_message("assistant"):
            st.markdown(assistant_message)

else:
    st.info("👆 Upload PDF files or enter a YouTube URL, then click 'Process Documents' to get started!")
