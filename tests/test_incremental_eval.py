import pytest
import os
import sys
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.eval.base import BenchmarkSample, EvidenceItem, BaseBenchmarkDataset
from app.eval.corpus_splitter import CorpusSplitter, CorpusSplit
from app.eval.incremental_reporter import IncrementalBenchmarkReporter
from app.eval.incremental_runner import IncrementalEvaluationRunner
from app.eval.indexer import BenchmarkIndexer
from app.services.embedding_service import EmbeddingService


def create_dummy_pdf(file_path: Path, title: str, pages_content: list):
    """Helper to create a small valid PDF using PyMuPDF (fitz)."""
    import pymupdf
    doc = pymupdf.open()
    for p_text in pages_content:
        page = doc.new_page()
        page.insert_text((50, 72), f"{title}\n\n{p_text}", fontsize=11)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(file_path))
    doc.close()


class DummySyntheticDataset(BaseBenchmarkDataset):
    """Synthetic dataset for hermetic, fast testing."""

    def __init__(self, samples: list, pdf_dir: Path):
        self.samples = samples
        self.pdf_dir = pdf_dir

    def load(self, n=None):
        if n is not None:
            return self.samples[:n]
        return self.samples

    def get_required_documents(self, samples):
        docs = set()
        for s in samples:
            if s.doc_name:
                docs.add(s.doc_name)
            for ev in s.evidence:
                if ev.doc_name:
                    docs.add(ev.doc_name)
        return sorted(list(docs))

    def get_pdf_path(self, doc_name: str) -> Path:
        filename = doc_name if doc_name.endswith(".pdf") else f"{doc_name}.pdf"
        return self.pdf_dir / filename


def test_corpus_splitter_partitioning():
    """Verify CorpusSplitter accurately partitions documents and categorizes queries into Base vs New."""
    samples = [
        BenchmarkSample(
            id="q1",
            question="What is revenue in DocA?",
            answer="$100M",
            doc_name="DocA",
            evidence=[EvidenceItem("Revenue was $100M", 0, "DocA")],
        ),
        BenchmarkSample(
            id="q2",
            question="What are expenses in DocB?",
            answer="$50M",
            doc_name="DocB",
            evidence=[EvidenceItem("Expenses were $50M", 0, "DocB")],
        ),
        BenchmarkSample(
            id="q3",
            question="What is net income in DocC?",
            answer="$20M",
            doc_name="DocC",
            evidence=[EvidenceItem("Net income was $20M", 0, "DocC")],
        ),
        BenchmarkSample(
            id="q4",
            question="What is capex in DocD?",
            answer="$10M",
            doc_name="DocD",
            evidence=[EvidenceItem("Capex was $10M", 0, "DocD")],
        ),
    ]

    # Split with base_fraction=0.5 (2 base docs: DocA, DocB; 2 update docs: DocC, DocD) and rounds=2
    split = CorpusSplitter.split(samples, base_fraction=0.5, rounds=2)

    assert split.all_docs == ["DocA", "DocB", "DocC", "DocD"]
    assert split.base_docs == ["DocA", "DocB"]
    assert split.update_docs == ["DocC", "DocD"]
    assert len(split.base_queries) == 2
    assert {s.id for s in split.base_queries} == {"q1", "q2"}
    assert len(split.new_queries) == 2
    assert {s.id for s in split.new_queries} == {"q3", "q4"}

    # Check multi-round partitions
    assert len(split.round_partitions) == 3  # Round 0 (base), Round 1, Round 2
    assert split.round_partitions[0].round_index == 0
    assert split.round_partitions[0].batch_docs == ["DocA", "DocB"]

    assert split.round_partitions[1].round_index == 1
    assert split.round_partitions[1].batch_docs == ["DocC"]
    assert [s.id for s in split.round_partitions[1].new_queries] == ["q3"]

    assert split.round_partitions[2].round_index == 2
    assert split.round_partitions[2].batch_docs == ["DocD"]
    assert [s.id for s in split.round_partitions[2].new_queries] == ["q4"]


def test_incremental_vs_full_rebuild_parity_and_efficiency(tmp_path):
    """
    End-to-end integration test:
    Ingests synthetic PDFs, compares Mode A (Incremental) against Mode B (Full Rebuild),
    and asserts efficiency savings and retrieval parity.
    """
    pdf_dir = tmp_path / "pdfs"
    bench_dir = tmp_path / "benchmark_runs"
    runs_dir = tmp_path / "runs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create 3 synthetic PDF documents
    create_dummy_pdf(
        pdf_dir / "AlphaCorp_2022.pdf",
        "AlphaCorp 2022 Financials",
        [
            "AlphaCorp Overview. AlphaCorp total revenue reached $500 million in FY2022.",
            "AlphaCorp Statement of Income. Operating income was $120 million.",
        ],
    )
    create_dummy_pdf(
        pdf_dir / "BetaTech_2022.pdf",
        "BetaTech 2022 Financials",
        [
            "BetaTech Overview. BetaTech total cloud ARR reached $300 million in FY2022.",
            "BetaTech Balance Sheet. Cash and cash equivalents stood at $80 million.",
        ],
    )
    create_dummy_pdf(
        pdf_dir / "GammaHoldings_2022.pdf",
        "GammaHoldings 2022 Financials",
        [
            "GammaHoldings Overview. Total assets under management reached $900 million in FY2022.",
            "GammaHoldings Cash Flow. Purchases of property and equipment capex was $45 million.",
        ],
    )

    # 2. Create 3 corresponding QA samples
    samples = [
        BenchmarkSample(
            id="s1_base_alpha",
            question="What is the FY2022 total revenue for AlphaCorp?",
            answer="$500 million",
            doc_name="AlphaCorp_2022",
            evidence=[
                EvidenceItem(
                    evidence_text="AlphaCorp total revenue reached $500 million",
                    evidence_page_num=0,  # Page 1
                    doc_name="AlphaCorp_2022",
                )
            ],
        ),
        BenchmarkSample(
            id="s2_base_beta",
            question="What was BetaTech cash balance in 2022?",
            answer="$80 million",
            doc_name="BetaTech_2022",
            evidence=[
                EvidenceItem(
                    evidence_text="Cash and cash equivalents stood at $80 million",
                    evidence_page_num=1,  # Page 2
                    doc_name="BetaTech_2022",
                )
            ],
        ),
        BenchmarkSample(
            id="s3_new_gamma",
            question="What was capex expenditure for GammaHoldings?",
            answer="$45 million",
            doc_name="GammaHoldings_2022",
            evidence=[
                EvidenceItem(
                    evidence_text="Purchases of property and equipment capex was $45 million",
                    evidence_page_num=1,  # Page 2
                    doc_name="GammaHoldings_2022",
                )
            ],
        ),
    ]

    mock_dataset = DummySyntheticDataset(samples=samples, pdf_dir=pdf_dir)

    # Mock GroqService
    mock_groq = MagicMock()
    mock_groq.generate_answer.side_effect = lambda prompt: (
        "AlphaCorp total revenue reached $500 million."
        if "AlphaCorp" in prompt
        else "BetaTech cash was $80 million."
        if "BetaTech" in prompt
        else "GammaHoldings capex was $45 million."
    )

    runner = IncrementalEvaluationRunner(
        dataset_name="financebench",
        data_dir=str(tmp_path),
        output_dir=str(bench_dir),
        base_fraction=0.66,  # 2 base docs (AlphaCorp, BetaTech), 1 update doc (GammaHoldings)
        rounds=1,
        top_k=3,
        judge_mode="semantic",
        groq_service=mock_groq,
        temp_dir=runs_dir,
        cleanup_temp=False,
    )
    # Override dataset instance with mock
    runner.dataset = mock_dataset

    # Run evaluation
    reporter = runner.run(n=3, skip_download=True, show_progress=False)

    summary = reporter.summary
    eff = summary["efficiency"]
    qp = summary["quality_parity"]

    # 3. Efficiency Assertions
    # In Mode A incremental update, only GammaHoldings (1 doc) was indexed in update step
    # In Mode B full rebuild, all 3 docs were indexed from scratch
    assert eff["incremental_update_embedding_calls"] < eff["full_rebuild_embedding_calls"], (
        "Incremental update should require fewer embedding calls than a full rebuild."
    )
    assert eff["embedding_calls_saved"] > 0
    assert eff["embedding_calls_saved_percentage"] > 0.0
    assert eff["chunk_reembedding_ratio"] < 1.0

    # 4. Parity Assertions
    # Mode A (Incremental) and Mode B (Full Rebuild) must achieve statistically equivalent retrieval
    base_delta = qp["base_queries"]["delta"]
    new_delta = qp["new_queries"]["delta"]
    overall_delta = qp["overall"]["delta"]

    # Retrieval Page Hit and Hit@5 parity (delta between -0.05 and +0.05)
    assert abs(base_delta["hit_at_5_delta"]) < 0.05
    assert abs(new_delta["hit_at_5_delta"]) < 0.05
    assert abs(overall_delta["hit_at_5_delta"]) < 0.05

    # 5. Output Artifacts Assertions
    json_files = list(bench_dir.glob("incremental_eval_*.json"))
    md_files = list(bench_dir.glob("incremental_summary_*.md"))

    assert len(json_files) == 1
    assert len(md_files) == 1

    with open(json_files[0], "r", encoding="utf-8") as f:
        json_data = json.load(f)
        assert "summary" in json_data
        assert "efficiency" in json_data["summary"]
        assert "quality_parity" in json_data["summary"]
        assert "samples" in json_data
        assert len(json_data["samples"]) == 3


def test_multi_round_stress_test(tmp_path):
    """Verify multi-round sequential updates track drift and accuracy across rounds."""
    pdf_dir = tmp_path / "pdfs"
    bench_dir = tmp_path / "benchmarks"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    create_dummy_pdf(pdf_dir / "Doc1.pdf", "Doc1 Title", ["Page 1 text for Doc1", "Page 2 text for Doc1"])
    create_dummy_pdf(pdf_dir / "Doc2.pdf", "Doc2 Title", ["Page 1 text for Doc2"])
    create_dummy_pdf(pdf_dir / "Doc3.pdf", "Doc3 Title", ["Page 1 text for Doc3"])

    samples = [
        BenchmarkSample(
            id="q1",
            question="Question on Doc1?",
            answer="Doc1 text",
            doc_name="Doc1",
            evidence=[EvidenceItem("Page 1 text for Doc1", 0, "Doc1")],
        ),
        BenchmarkSample(
            id="q2",
            question="Question on Doc2?",
            answer="Doc2 text",
            doc_name="Doc2",
            evidence=[EvidenceItem("Page 1 text for Doc2", 0, "Doc2")],
        ),
        BenchmarkSample(
            id="q3",
            question="Question on Doc3?",
            answer="Doc3 text",
            doc_name="Doc3",
            evidence=[EvidenceItem("Page 1 text for Doc3", 0, "Doc3")],
        ),
    ]

    mock_dataset = DummySyntheticDataset(samples=samples, pdf_dir=pdf_dir)

    runner = IncrementalEvaluationRunner(
        dataset_name="financebench",
        data_dir=str(tmp_path),
        output_dir=str(bench_dir),
        base_fraction=0.34,  # Doc1 base, Doc2 update 1, Doc3 update 2
        rounds=2,
        top_k=3,
        judge_mode="semantic",
        groq_service=None,  # Pure semantic evaluation
        temp_dir=tmp_path / "runs",
        cleanup_temp=True,
    )
    runner.dataset = mock_dataset

    reporter = runner.run(show_progress=False)

    stress_stats = reporter.round_stress_test_stats
    assert len(stress_stats) >= 3  # Round 0, Round 1, Round 2
    assert stress_stats[0]["round_index"] == 0
    assert stress_stats[1]["round_index"] == 1
    assert stress_stats[2]["round_index"] == 2
    assert "drift_from_round_0" in stress_stats[1]
