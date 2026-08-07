# Financial Incremental RAG

## Overview

Financial Incremental RAG is a Financial Document Intelligence System that enables users to query financial reports using Retrieval-Augmented Generation (RAG). The system introduces chunk-level incremental indexing, allowing only modified document sections to be re-embedded and updated without rebuilding the entire vector database.

## Features

- Upload financial documents (PDF, DOCX, TXT)
- Semantic document chunking
- Financial question answering using Groq LLM
- ChromaDB vector storage
- Chunk-level incremental indexing
- Performance dashboard
- FastAPI backend
- Streamlit frontend

## Tech Stack

- Python
- FastAPI
- Streamlit
- Groq API
- Sentence Transformers
- ChromaDB
- SQLite
- LangChain
- PyMuPDF

## Project Status

🚧 MVP Development In Progress
