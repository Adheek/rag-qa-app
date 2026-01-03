# Academic QA - RAG System

An intelligent question-answering system built with Retrieval-Augmented Generation (RAG) using LangChain, LangGraph, and Groq LLM.

## Features

- 📄 **PDF Upload**: Upload and process multiple PDF documents
- 🎥 **YouTube Integration**: Extract and process YouTube video transcripts
- 💬 **Interactive Chat**: Ask questions about your uploaded content
- 🧠 **Context-Aware**: Maintains conversation history for follow-up questions
- ⚡ **Fast Processing**: Uses efficient embedding models and vector search

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (llama-3.1-8b-instant)
- **Framework**: LangChain + LangGraph
- **Vector Store**: ChromaDB
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload PDF files or paste a YouTube URL
2. Click "Process Documents" to create the knowledge base
3. Ask questions in the chat interface
4. The system will retrieve relevant context and generate answers

## License

MIT
