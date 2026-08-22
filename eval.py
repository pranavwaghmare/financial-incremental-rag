#!/usr/bin/env python
"""
Financial Incremental RAG - Benchmark Evaluation CLI
Benchmark the RAG system against standard financial QA datasets such as FinanceBench.

Usage:
  python eval.py --dataset financebench --n 20
  python eval.py --dataset financebench
  python eval.py --dataset financebench --judge semantic --top-k 3
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.eval.runner import EvaluationRunner, DATASET_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Financial Incremental RAG on financial QA datasets (e.g. FinanceBench)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="financebench",
        choices=list(DATASET_REGISTRY.keys()),
        help="Benchmark dataset to evaluate on (default: financebench)",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all questions in the dataset)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks per query (default: 5)",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="hybrid",
        choices=["hybrid", "semantic", "llm", "numeric"],
        help="Answer evaluation judge mode (default: hybrid)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_runs",
        help="Directory to save JSON and Markdown benchmark reports (default: benchmark_runs)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/financebench",
        help="Directory for dataset cache and downloaded PDFs (default: data/financebench)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="financebench_docs",
        help="ChromaDB collection name for benchmark documents (default: financebench_docs)",
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default="chroma_storage",
        help="Path to ChromaDB storage directory (default: chroma_storage)",
    )
    parser.add_argument(
        "--manifest-db",
        type=str,
        default="manifest_db/manifest.db",
        help="Path to SQLite manifest database (default: manifest_db/manifest.db)",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip document download and indexing step (assumes docs are already indexed)",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only download and index benchmark PDFs without evaluating queries",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    runner = EvaluationRunner(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        manifest_db_path=args.manifest_db,
        chroma_path=args.chroma_path,
        chroma_collection_name=args.collection,
        top_k=args.top_k,
        judge_mode=args.judge,
    )

    if args.index_only:
        print(f"[*] Running Index-Only mode for '{args.dataset}'...")
        samples = runner.dataset.load(n=args.n)
        pdf_paths = runner.dataset.download_pdfs_for_samples(samples)
        stats = runner.indexer.index_documents(pdf_paths, show_progress=True)
        print(f"[+] Indexing complete: {stats['total_chunks']} chunks in collection '{args.collection}'.")
        return

    reporter = runner.run(
        n=args.n,
        skip_ingestion=args.skip_ingestion,
        show_progress=True,
    )


if __name__ == "__main__":
    main()
