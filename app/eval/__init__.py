"""
Evaluation and benchmarking suite for Financial Incremental RAG.
"""

from app.eval.base import (
    BenchmarkSample,
    EvidenceItem,
    RetrievalScore,
    AnswerScore,
    SampleEvaluationResult,
    BaseBenchmarkDataset,
)
from app.eval.financebench_loader import FinanceBenchDataset
from app.eval.indexer import BenchmarkIndexer
from app.eval.metrics import RetrievalEvaluator, AnswerEvaluator
from app.eval.reporter import BenchmarkReporter
from app.eval.corpus_splitter import CorpusSplitter, CorpusSplit, RoundPartition
from app.eval.incremental_reporter import IncrementalBenchmarkReporter
from app.eval.incremental_runner import IncrementalEvaluationRunner

__all__ = [
    "BenchmarkSample",
    "EvidenceItem",
    "RetrievalScore",
    "AnswerScore",
    "SampleEvaluationResult",
    "BaseBenchmarkDataset",
    "FinanceBenchDataset",
    "BenchmarkIndexer",
    "RetrievalEvaluator",
    "AnswerEvaluator",
    "BenchmarkReporter",
    "EvaluationRunner",
    "DATASET_REGISTRY",
    "CorpusSplitter",
    "CorpusSplit",
    "RoundPartition",
    "IncrementalBenchmarkReporter",
    "IncrementalEvaluationRunner",
]

