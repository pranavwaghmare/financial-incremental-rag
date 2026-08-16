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
- Next.js frontend

## Tech Stack

- Python
- FastAPI
- Next.js (React)
- Groq API
- Sentence Transformers
- ChromaDB
- SQLite
- LangChain
- PyMuPDF

## Why Incremental RAG?

Traditional RAG setups often rebuild the entire document index or re-embed large sections even for minor modifications. **Incremental RAG** solves this by leveraging chunk-level indexing. When a document is modified, the system:
1. Identifies exactly which chunks have changed.
2. Re-embeds and updates only those specific chunks in the vector database.
3. Preserves unmodified chunks, drastically reducing compute time, API costs, and embedding latency.
This makes the system significantly faster and more cost-effective for constantly evolving financial datasets.

## How to Run the Project

The project is split into a Python/FastAPI backend and a Next.js frontend.

### 1. Backend (FastAPI)
1. Ensure your virtual environment is activated (e.g., `.venv\Scripts\activate`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```
   The API will run at `http://localhost:8000`.

### 2. Frontend (Next.js)
1. Navigate to the `web` directory:
   ```bash
   cd web
   ```
2. Install node modules (first time only):
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
   The frontend will run at `http://localhost:3000`.

## Local Data and Generated Files

This project creates local runtime data such as:
- ChromaDB storage in `chroma_storage/`
- SQLite manifests in `manifest_db/`
- analytics data in `analytics_db/`
- uploaded PDFs in `documents/`
- benchmark output in `benchmark_runs/`

These files are generated locally and are intentionally ignored in `.gitignore`.

## How to Run Tests

The backend uses `pytest` for testing. To run the test suite:

1. Ensure your virtual environment is activated.
2. Run `pytest` from the root directory:
   ```bash
   pytest
   ```
   For detailed test output, run:
   ```bash
   pytest -v
   ```

## Project Status

🚧 MVP Development In Progress
