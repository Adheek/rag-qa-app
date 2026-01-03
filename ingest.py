import os

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Updated import

# Use the newer HuggingFace embeddings package
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from config import config

def load_youtube_content(url: str, save_dir: str):
    """
    Loads audio from a YouTube URL, transcribes it using FasterWhisperParser,
    and returns the loaded documents.
    
    Args:
        url (str): The URL of the YouTube video.
        save_dir (str): Directory to save the audio files temporarily.
    
    Returns:
        list: A list of documents loaded from the YouTube content.
    """
    print(f"Starting YouTube content loading from: {url}")
    try:
        loader = GenericLoader(
            YoutubeAudioLoader([url], save_dir),
            FasterWhisperParser()
        )
        docs = loader.load()
        print(f"Successfully loaded {len(docs)} documents from YouTube.")
        return docs
    except Exception as e:
        print(f"Error loading YouTube content: {e}")
        return []

def load_pdf_content(pdf_directory: str):
    """
    Loads all PDF files from the specified directory.
    
    Args:
        pdf_directory (str): Path to the directory containing PDF files.
    
    Returns:
        list: A list of documents loaded from PDF files.
    """
    # Ensure the PDF directory exists
    if not os.path.exists(pdf_directory):
        print(f"Error: PDF directory '{pdf_directory}' not found.")
        print(f"Please create this directory and place your PDF files inside.")
        return []
    
    all_pdf_docs = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            print(f"Loading PDF document: {filepath}")
            try:
                loader = PyPDFLoader(filepath)
                pages = loader.load()
                all_pdf_docs.extend(pages)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    if not all_pdf_docs:
        print("No PDF documents found or loaded in the specified directory.")
    else:
        print(f"Loaded {len(all_pdf_docs)} pages from PDF documents.")
    
    return all_pdf_docs

def ingest_all_documents(
    youtube_url: str,
    youtube_save_dir: str,
    pdf_directory: str = "data",
    embeddings_model=None,
    persist_directory: str = "docs/chroma"
):
    """
    Loads documents from YouTube and PDF sources, splits them into chunks,
    generates embeddings, and stores them in ChromaDB.
    
    Args:
        youtube_url (str): URL of the YouTube video.
        youtube_save_dir (str): Directory to save YouTube audio temporarily.
        pdf_directory (str): Directory containing PDF files.
        embeddings_model: The embeddings model to use.
        persist_directory (str): Directory to persist the ChromaDB.
    
    Returns:
        Chroma: The ChromaDB vector store with ingested documents.
    """
    # 1. Load YouTube content
    youtube_docs = load_youtube_content(youtube_url, youtube_save_dir)
    
    # 2. Load PDF content
    pdf_docs = load_pdf_content(pdf_directory)
    
    # Combine all loaded documents
    combined_docs = youtube_docs + pdf_docs
    if not combined_docs:
        print("No documents (PDF or YouTube) were loaded. Exiting ingestion.")
        return
    
    print(f"\nTotal combined documents loaded: {len(combined_docs)}")
    
    # 3. Document Splitters
    chunk_size = config.CHUNK_SIZE
    chunk_overlap = config.CHUNK_OVERLAP
    
    print(f"Splitting documents into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs = r_splitter.split_documents(combined_docs)
    print(f"Split documents into {len(chunked_docs)} chunks.")
    
    # 4. Embeddings
    model_name = config.EMBEDDING_MODEL_NAME
    
    print(f"Initializing embeddings with model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # Force CPU to avoid PyTorch issues
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 5. Create and persist the Chroma vector store
    print(f"Creating ChromaDB vector store at '{persist_directory}'...")
    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"Vector store created and persisted successfully at '{persist_directory}'.")
    print(f"Total documents in vector store: {vectordb._collection.count()}")
    
    return vectordb

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Document Ingestion Process")
    print("=" * 60)
    
    # Use configuration from config.py
    youtube_url = config.YOUTUBE_VIDEO_URL
    youtube_save_dir = config.YOUTUBE_AUDIO_SAVE_DIRECTORY
    pdf_directory = config.PDF_SOURCE_DIRECTORY
    persist_directory = config.CHROMA_PERSIST_DIRECTORY
    
    # Run the ingestion
    vectordb = ingest_all_documents(
        youtube_url=youtube_url,
        youtube_save_dir=youtube_save_dir,
        pdf_directory=pdf_directory,
        persist_directory=persist_directory
    )
    
    if vectordb:
        print("\n" + "=" * 60)
        print("Ingestion Complete!")
        print("=" * 60)
        print(f"You can now run the Streamlit app with: streamlit run app.py")
    else:
        print("\nIngestion failed. Please check the errors above.")