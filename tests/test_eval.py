import pytest
import os
import sys
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.eval.base import BenchmarkSample, EvidenceItem, RetrievalScore, AnswerScore
from app.eval.financebench_loader import FinanceBenchDataset
from app.eval.metrics import RetrievalEvaluator, AnswerEvaluator
from app.eval.reporter import BenchmarkReporter
from app.eval.runner import EvaluationRunner
from app.eval.indexer import BenchmarkIndexer
from app.services.embedding_service import EmbeddingService


def create_dummy_pdf(file_path: Path, title: str, paragraphs: list):
    """Helper to create a small valid PDF using PyMuPDF (fitz)."""
    import pymupdf
    doc = pymupdf.open()
    for p_text in paragraphs:
        page = doc.new_page()
        page.insert_text((50, 72), f"{title}\n\n{p_text}", fontsize=11)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(file_path))
    doc.close()


def test_financebench_evidence_page_indexing():
    """Verify that 0-indexed benchmark pages correctly convert to 1-indexed PDF pages."""
    ev = EvidenceItem(
        evidence_text="Capital expenditures: $1,577 million",
        evidence_page_num=59,  # 0-indexed
        doc_name="3M_2018_10K"
    )
    assert ev.page_number_1_indexed == 60


def test_retrieval_evaluator():
    """Test retrieval metrics calculation (Hit@k, MRR, Doc Hit, Page Hit)."""
    sample = BenchmarkSample(
        id="fb_test_1",
        question="What is the capex for 3M in 2018?",
        answer="$1577.00",
        doc_name="3M_2018_10K",
        evidence=[
            EvidenceItem(
                evidence_text="Purchases of property, plant and equipment $1,577",
                evidence_page_num=59,  # 0-indexed -> page 60
                doc_name="3M_2018_10K",
            )
        ],
    )

    # Chunks where rank 2 has the correct page (page 60 of 3M_2018_10K)
    retrieved_chunks = [
        {"document": "3M_2018_10K.pdf", "page": 12, "text": "Overview of operations"},
        {"document": "3M_2018_10K.pdf", "page": 60, "text": "Purchases of property, plant and equipment $1,577"},
        {"document": "OTHER_2018.pdf", "page": 60, "text": "Irrelevant text"},
    ]

    score = RetrievalEvaluator.evaluate_retrieval(retrieved_chunks, sample, top_k=3)

    assert score.doc_hit is True
    assert score.page_hit is True
    assert score.hit_at_1 is False
    assert score.hit_at_3 is True
    assert score.hit_at_5 is True
    assert score.mrr == pytest.approx(0.5, 0.01)
    assert score.text_overlap_score > 0.5


def test_answer_evaluator_numeric_and_exact():
    """Test numeric matching and exact matching for financial answers."""
    evaluator = AnswerEvaluator(embedder=EmbeddingService())

    # Direct numeric match
    assert evaluator.check_numeric_match(
        generated_answer="The FY2018 capital expenditure for 3M was $1,577.00 million.",
        gold_answer="$1577.00",
    )

    # Percentage matching
    assert evaluator.check_numeric_match(
        generated_answer="The quick ratio was calculated as 0.96 as of June 2023.",
        gold_answer="0.96",
    )

    # Negative number in parentheses format (accounting)
    assert evaluator.check_numeric_match(
        generated_answer="Operating income changed by -1.7% due to one-off costs.",
        gold_answer="Operating Margin has decreased by 1.7%",
    )

    # Exact string matching
    assert evaluator.check_exact_match(
        generated_answer="The consumer segment shrunk by 0.9% organically.",
        gold_answer="The consumer segment shrunk by 0.9% organically.",
    )

    # Semantic similarity
    sim = evaluator.compute_semantic_similarity(
        "3M maintains consecutive dividend increases for 65 years.",
        "Yes, 3M has been increasing dividend for 65 consecutive years.",
    )
    assert sim >= 0.80


def test_eval_runner_tiny_subset(tmp_path):
    """End-to-end regression test: runs evaluation on 3 mock questions with local PDFs and mocked LLM."""
    test_data_dir = tmp_path / "data"
    test_pdf_dir = test_data_dir / "pdfs"
    test_bench_dir = tmp_path / "benchmark_runs"
    manifest_path = tmp_path / "manifest.db"
    chroma_path = tmp_path / "chroma"

    test_pdf_dir.mkdir(parents=True, exist_ok=True)
    test_bench_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy PDF for 3M_2018_10K with 3 pages
    pdf_file = test_pdf_dir / "3M_2018_10K.pdf"
    create_dummy_pdf(
        pdf_file,
        title="3M 2018 Annual Report",
        paragraphs=[
            "Page 1: General company overview and highlights.",
            "Page 2: Balance Sheet with Cash and assets.",
            "Page 3: Consolidated Statement of Cash Flows. Purchases of property, plant and equipment was $1,577 million.",
        ],
    )

    # Create 3 synthetic questions matching FinanceBench structure
    test_jsonl = test_data_dir / "financebench_open_source.jsonl"
    sample_records = [
        {
            "financebench_id": "test_01",
            "company": "3M",
            "doc_name": "3M_2018_10K",
            "question": "What is the FY2018 capital expenditure for 3M?",
            "answer": "$1577.00",
            "justification": "Extracted from Cash Flows.",
            "evidence": [
                {
                    "evidence_text": "Purchases of property, plant and equipment was $1,577 million",
                    "evidence_page_num": 2,  # 0-indexed page 2 -> 1-indexed page 3
                    "doc_name": "3M_2018_10K",
                }
            ],
        },
        {
            "financebench_id": "test_02",
            "company": "3M",
            "doc_name": "3M_2018_10K",
            "question": "What is on page 1 of 3M report?",
            "answer": "General company overview and highlights.",
            "justification": "From page 1",
            "evidence": [
                {
                    "evidence_text": "General company overview and highlights.",
                    "evidence_page_num": 0,  # 0-indexed -> page 1
                    "doc_name": "3M_2018_10K",
                }
            ],
        },
        {
            "financebench_id": "test_03",
            "company": "3M",
            "doc_name": "3M_2018_10K",
            "question": "Where is the balance sheet located?",
            "answer": "Page 2 contains Balance Sheet with Cash and assets.",
            "justification": "From page 2",
            "evidence": [
                {
                    "evidence_text": "Balance Sheet with Cash and assets.",
                    "evidence_page_num": 1,  # 0-indexed -> page 2
                    "doc_name": "3M_2018_10K",
                }
            ],
        },
    ]

    with open(test_jsonl, "w", encoding="utf-8") as f:
        for r in sample_records:
            f.write(json.dumps(r) + "\n")

    # Mock GroqService to avoid external API calls in CI
    mock_groq = MagicMock()
    mock_groq.generate_answer.side_effect = lambda prompt: (
        "The FY2018 capital expenditure for 3M was $1,577.00 million."
        if "capital expenditure" in prompt
        else "General company overview and highlights."
    )

    # Initialize runner with test paths
    runner = EvaluationRunner(
        dataset_name="financebench",
        data_dir=str(test_data_dir),
        output_dir=str(test_bench_dir),
        manifest_db_path=str(manifest_path),
        chroma_path=str(chroma_path),
        chroma_collection_name="test_financebench",
        top_k=3,
        judge_mode="semantic",
        groq_service=mock_groq,
    )

    # Run evaluation on the 3 questions
    reporter = runner.run(n=3, skip_ingestion=False, show_progress=False)

    # Verify summary
    assert reporter.summary["total_samples"] == 3
    assert "retrieval" in reporter.summary
    assert "answer_quality" in reporter.summary
    assert reporter.summary["retrieval"]["page_hit_rate"] > 0.0
    assert reporter.summary["answer_quality"]["accuracy"] > 0.0

    # Verify files generated in benchmark_runs
    json_files = list(test_bench_dir.glob("financebench_eval_*.json"))
    md_files = list(test_bench_dir.glob("financebench_summary_*.md"))

    assert len(json_files) == 1
    assert len(md_files) == 1

    # Verify JSON content structure
    with open(json_files[0], "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        assert "summary" in saved_data
        assert "samples" in saved_data
        assert len(saved_data["samples"]) == 3
        assert saved_data["samples"][0]["id"] == "test_01"
