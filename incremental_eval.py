#!/usr/bin/env python
"""
Financial Incremental RAG - Incremental Indexing Efficiency & Parity Evaluation CLI

Evaluates the efficiency and quality parity of chunk-level incremental indexing
against a full-rebuild-from-scratch baseline on financial QA datasets (e.g. FinanceBench),
following the evaluation methodology in:
- ECT-QA / 'RAG Meets Temporal Graphs' (arXiv:2510.13590): Base/Update corpus and Base/New queries
- EraRAG (arXiv:2506.20963): Incremental vs. Full Rebuild efficiency & parity benchmarks

Usage:
  # Quick 10-question evaluation across 2 incremental rounds
  python incremental_eval.py --n 10 --rounds 2

  # 20 questions with 80% base corpus and 3 incremental update rounds
  python incremental_eval.py --n 20 --base-fraction 0.8 --rounds 3

  # Full dataset evaluation with hybrid answer evaluation
  python incremental_eval.py --judge hybrid --rounds 3
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

from app.eval.runner import DATASET_REGISTRY
from app.eval.incremental_runner import IncrementalEvaluationRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Incremental Indexing Efficiency and Parity for Financial Incremental RAG "
            "using ECT-QA and EraRAG research methodology."
        )
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
        default=20,
        help="Number of questions to evaluate (default: 20; pass 0 or omit with --all for full dataset)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate on the full dataset (overrides -n)",
    )
    parser.add_argument(
        "--base-fraction",
        type=float,
        default=0.8,
        help="Fraction of documents assigned to the base index (default: 0.8 = 80%% base, 20%% update)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of sequential incremental update rounds / batches (default: 3)",
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
        default="semantic",
        choices=["semantic", "hybrid", "llm", "numeric"],
        help="Answer evaluation judge mode (default: semantic [zero API cost]; 'hybrid' for LLM+semantic)",
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
        "--skip-download",
        action="store_true",
        help="Skip downloading PDFs if already present in data directory",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Retain isolated temporary Chroma and SQLite databases in .runs/ for debugging",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sample_count = None if args.all or (args.n is not None and args.n <= 0) else args.n

    runner = IncrementalEvaluationRunner(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_fraction=args.base_fraction,
        rounds=args.rounds,
        top_k=args.top_k,
        judge_mode=args.judge,
        cleanup_temp=not args.no_cleanup,
    )

    reporter = runner.run(
        n=sample_count,
        skip_download=args.skip_download,
        show_progress=True,
    )


if __name__ == "__main__":
    main()
