import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


class Config:
    """Configuration class for RAG application."""

    # ─── Embedding Model Configuration ───
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 2028
    CHUNK_OVERLAP = 250


def process_pdf_files(uploaded_files):
    """Process uploaded PDF files and return documents."""
    all_docs = []
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        all_docs.extend(docs)

        # Clean up temp file
        os.remove(temp_path)

    return all_docs


def process_youtube_url(youtube_url):
    """Process YouTube URL with FasterWhisper (same as your notebook)."""
    save_dir = "docs/youtube"
    os.makedirs(save_dir, exist_ok=True)

    loader = GenericLoader(
        YoutubeAudioLoader([youtube_url], save_dir),
        FasterWhisperParser()
    )
    docs = loader.load()
    return docs


def create_vector_store(documents, embeddings):
    """Create ChromaDB vector store from documents."""
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)

    # Create vector store
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectordb


def get_embeddings_model():
    """Initialize embeddings model."""
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


# Create global config instance
config = Config()
