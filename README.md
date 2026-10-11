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
   For detailed test output:
   ```bash
   pytest -v
   ```
   To run only the evaluation suite:
   ```bash
   pytest -v tests/test_eval.py
   ```

## Benchmarking

This project includes an automated evaluation harness in `eval.py` (powered by `app.eval`) to benchmark retrieval and answer quality against [**FinanceBench**](https://github.com/patronus-ai/financebench), the financial question-answering benchmark from Patronus AI (150 open-source QA pairs across 10-K, 10-Q, and 8-K filings).

### 1. Running the Benchmark

```bash
# Run on a 20-question sample
python eval.py --dataset financebench --n 20

# Run on the full 150-question open-source sample
python eval.py --dataset financebench

# Run with local semantic similarity only (zero LLM API cost)
python eval.py --dataset financebench --n 20 --judge semantic

# Ingest and index PDFs without running evaluation queries
python eval.py --dataset financebench --index-only
```

### 2. Available CLI Arguments

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | Benchmark dataset (`financebench`) | `financebench` |
| `-n`, `--n` | Number of questions to evaluate | All (150) |
| `--top-k` | Number of chunks retrieved per query | `5` |
| `--judge` | Answer evaluation mode (`hybrid`, `semantic`, `llm`, `numeric`) | `hybrid` |
| `--output-dir`| Directory for benchmark run outputs | `benchmark_runs` |
| `--data-dir` | Directory for downloaded dataset & PDF cache | `data/financebench` |
| `--collection`| ChromaDB collection name for benchmark documents | `financebench_docs` |
| `--skip-ingestion` | Skip PDF download and indexing step | `False` |
| `--index-only` | Download & index PDFs without running queries | `False` |

### 3. Metrics Evaluated

- **Retrieval Performance**:
  - `Hit@1`, `Hit@3`, `Hit@5`: Page & document match in top-k retrieved chunks.
  - `Page Hit Rate`: Whether the retrieved chunks include the gold target page.
  - `Document Hit Rate`: Whether the retrieved chunks include the target SEC filing.
  - `MRR (Mean Reciprocal Rank)`: Position of the first correct chunk.
  - `Text Overlap`: Token and character overlap with the ground-truth evidence text.
- **Answer Quality**:
  - `Exact Match`: Normalized string match with gold answer.
  - `Numeric Match`: Financial number extraction comparing dollar amounts, percentages, negative numbers, and unit scalings within tolerance.
  - `Semantic Similarity`: Embedding cosine similarity using SentenceTransformers (`all-MiniLM-L6-v2`).
  - `LLM-as-a-Judge`: Groq-powered factual correctness scoring against gold answers and justifications.

### 4. Benchmark Outputs

Reports are automatically generated and saved to `benchmark_runs/`:
- **JSON Report**: `benchmark_runs/financebench_eval_<timestamp>.json` (contains full aggregate metrics and sample-by-sample details).
- **Markdown Summary**: `benchmark_runs/financebench_summary_<timestamp>.md` (human-readable report with metric tables and sample breakdown).

---

## Incremental Evaluation Harness (ECT-QA & EraRAG Methodology)

To rigorously evaluate the system's core differentiator (**chunk-level incremental indexing**), we provide an incremental evaluation harness in `incremental_eval.py`. This follows the evaluation methodology of two research papers:
1. **ECT-QA / "RAG Meets Temporal Graphs" (arXiv:2510.13590)**: Splits the corpus into a base slice and an update slice, and independently measures retrieval & accuracy on **base queries** vs. **new queries**.
2. **EraRAG (arXiv:2506.20963)**: For every incremental update, benchmarks the incremental pipeline (**Mode A**) against a full-rebuild-from-scratch baseline (**Mode B**), comparing wall-clock index build time, embedding calls, and retrieval quality parity ($\Delta \text{accuracy} = \text{acc}_{\text{inc}} - \text{acc}_{\text{full}}$).

### 1. Running the Incremental Benchmark

```bash
# Quick run with 20 questions, 80% base corpus, across 3 sequential update rounds
python incremental_eval.py --n 20 --base-fraction 0.8 --rounds 3

# Fast 10-question evaluation with local semantic judge (zero LLM API cost)
python incremental_eval.py --n 10 --judge semantic

# Multi-round stress test across 5 sequential batches
python incremental_eval.py --rounds 5 --base-fraction 0.7
```

### 2. Available CLI Arguments

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | Benchmark dataset (`financebench`) | `financebench` |
| `-n`, `--n` | Number of questions to evaluate | `20` |
| `--all` | Evaluate on the full dataset | `False` |
| `--base-fraction` | Fraction of documents in the base index | `0.8` (80% base, 20% update) |
| `--rounds` | Number of sequential incremental update batches | `3` |
| `--top-k` | Number of chunks retrieved per query | `5` |
| `--judge` | Answer evaluation judge mode (`semantic`, `hybrid`, `llm`, `numeric`) | `semantic` |
| `--output-dir` | Directory for benchmark outputs | `benchmark_runs` |
| `--skip-download` | Skip downloading PDFs if already cached | `False` |
| `--no-cleanup` | Retain intermediate isolated Chroma & SQLite DBs in `.runs/` | `False` |

### 3. Key Incremental Metrics Captured

- **Efficiency Gains**:
  - `Wall-Clock Time Saved` & `Speedup Factor`: Time to incrementally update vs. rebuilding entire vector store from scratch.
  - `Embedding Calls Saved`: Exact number and percentage of embeddings skipped due to chunk-level manifest matching.
  - `Chunk Re-embedded Ratio`: Ratio of chunks re-embedded in the update step vs. total chunks in the corpus.
- **Quality Parity ($\Delta = \text{Mode A} - \text{Mode B}$)**:
  - Base query & new query Hit@1, Hit@3, Hit@5, Page Hit Rate, MRR, and Answer Accuracy deltas (empirically proving that incremental indexing does not sacrifice retrieval quality).
- **Multi-Round Drift / Stress Testing**:
  - Tracks retrieval performance on base queries as sequential batches of new documents are added to the database.

### 4. Incremental Benchmark Outputs

- **JSON Report**: `benchmark_runs/incremental_eval_<timestamp>.json`
- **Markdown Summary**: `benchmark_runs/incremental_summary_<timestamp>.md`

## Project Status

🚧 MVP Development In Progress

## Recommended Demonstration Flow

The clearest demonstration is to upload the same report twice: first the original
version, then a revised version with one section inserted or edited.

1. Start the backend and frontend.
2. Upload the original PDF from the Upload Report screen.
3. Note the initial embedding count.
4. Insert or edit one section while keeping the filename unchanged.
5. Upload the revised PDF.
6. Show the result panel: unchanged chunks are reused, only changed content is
   embedded, and the saved embedding count is visible.
7. Ask a question about both an unchanged section and the edited section to show
   that retrieval still works after the update.

For the benchmark evidence, run:

```bash
python incremental_eval.py --n 20 --base-fraction 0.8 --rounds 3 --judge semantic
```

The generated report compares incremental indexing with a full rebuild using
embedding calls saved, re-embedding ratio, indexing time, speedup, and retrieval
quality parity. Chunk IDs are content-addressable, so moving unchanged content
within a document does not change its vector identity.

