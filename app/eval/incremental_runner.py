import time
import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from tqdm import tqdm

from app.eval.base import (
    BaseBenchmarkDataset,
    BenchmarkSample,
    SampleEvaluationResult,
    RetrievalScore,
    AnswerScore,
)
from app.eval.financebench_loader import FinanceBenchDataset
from app.eval.corpus_splitter import CorpusSplitter, CorpusSplit, RoundPartition
from app.eval.indexer import BenchmarkIndexer
from app.eval.metrics import RetrievalEvaluator, AnswerEvaluator
from app.eval.incremental_reporter import IncrementalBenchmarkReporter
from app.eval.runner import DATASET_REGISTRY

from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService
from app.services.groq_service import GroqService
from app.services.embedding_service import EmbeddingService
from app.database.chroma_db import ChromaDBClient


class IncrementalEvaluationRunner:
    """
    Orchestrates the incremental efficiency and quality parity evaluation harness:
    1. Splits corpus into Base vs Update documents and Base vs New queries (ECT-QA).
    2. Runs Mode A: Incremental Indexing (Base index + sequential incremental batches).
    3. Runs Mode B: Full Rebuild Baseline (Clean index built from scratch).
    4. Measures efficiency gains (time speedup, embedding calls saved, chunk re-embedding ratio).
    5. Measures quality parity (Hit@k, Page Hit, MRR, Answer Accuracy deltas).
    6. Measures multi-round drift on Base queries across sequential update rounds.
    """

    def __init__(
        self,
        dataset_name: str = "financebench",
        data_dir: str = "data/financebench",
        output_dir: str = "benchmark_runs",
        base_fraction: float = 0.8,
        rounds: int = 1,
        top_k: int = 5,
        judge_mode: str = "semantic",
        groq_service: Optional[GroqService] = None,
        temp_dir: Optional[Union[str, Path]] = None,
        cleanup_temp: bool = True,
    ):
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.base_fraction = base_fraction
        self.rounds = rounds
        self.top_k = top_k
        self.judge_mode = judge_mode
        self.temp_dir = Path(temp_dir) if temp_dir else None
        self.cleanup_temp = cleanup_temp

        # Initialize dataset loader
        dataset_cls = DATASET_REGISTRY.get(self.dataset_name)
        if not dataset_cls:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}"
            )
        self.dataset: BaseBenchmarkDataset = dataset_cls(data_dir=str(self.data_dir))

        # Embedding & Groq services
        self.embedder = EmbeddingService()
        if groq_service is not None:
            self.groq_service = groq_service
        elif self.judge_mode in ("llm", "hybrid"):
            try:
                self.groq_service = GroqService()
            except Exception as e:
                self.groq_service = None
                if self.judge_mode in ("llm", "hybrid"):
                    self.judge_mode = "semantic"
        else:
            self.groq_service = None

        self.answer_evaluator = AnswerEvaluator(
            embedder=self.embedder,
            groq_service=self.groq_service,
        )

    def _evaluate_queries(
        self,
        samples: List[BenchmarkSample],
        retriever: RetrievalService,
        rag_service: RAGService,
        desc: str = "Evaluating",
        show_progress: bool = True,
    ) -> List[SampleEvaluationResult]:
        """Evaluate a list of benchmark samples on a specific retriever/rag_service."""
        results: List[SampleEvaluationResult] = []
        iterator = tqdm(samples, desc=desc, leave=False) if show_progress else samples

        for sample in iterator:
            q_start = time.perf_counter()

            if self.groq_service:
                rag_out = rag_service.ask(sample.question, top_k=self.top_k, log_analytics=False)
                generated_answer = rag_out.get("answer", "")
                retrieved_chunks = rag_out.get("sources", [])
            else:
                retrieved_chunks = retriever.retrieve(sample.question, top_k=self.top_k)
                generated_answer = (
                    retrieved_chunks[0].get("text", "")[:100]
                    if retrieved_chunks
                    else "No context retrieved"
                )

            q_latency = round(time.perf_counter() - q_start, 3)

            ret_score = RetrievalEvaluator.evaluate_retrieval(
                retrieved_chunks=retrieved_chunks,
                sample=sample,
                top_k=self.top_k,
            )

            ans_score = self.answer_evaluator.evaluate(
                generated_answer=generated_answer,
                sample=sample,
                mode=self.judge_mode,
            )

            results.append(
                SampleEvaluationResult(
                    sample=sample,
                    generated_answer=generated_answer,
                    retrieved_chunks=retrieved_chunks,
                    retrieval_score=ret_score,
                    answer_score=ans_score,
                    latency_seconds=q_latency,
                )
            )

        return results

    def run(
        self,
        n: Optional[int] = None,
        skip_download: bool = False,
        show_progress: bool = True,
    ) -> IncrementalBenchmarkReporter:
        """Run the full incremental evaluation workflow."""
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        print(f"\n{'='*75}")
        print(f"[*] Starting Incremental RAG Evaluation on '{self.dataset_name}'")
        print(f"    - Base Fraction: {self.base_fraction*100:.0f}%")
        print(f"    - Incremental Rounds: {self.rounds}")
        print(f"    - Sample Subset: {n or 'all'}")
        print(f"    - Judge Mode: {self.judge_mode}")
        print(f"{'='*75}")

        # 1. Load Samples
        samples = self.dataset.load(n=n)
        print(f"[*] Loaded {len(samples)} benchmark samples from {self.dataset_name}.")

        # 2. Download / Cache Required PDFs
        pdf_paths: List[Path] = []
        if isinstance(self.dataset, FinanceBenchDataset) and not skip_download:
            pdf_paths = self.dataset.download_pdfs_for_samples(samples)
        else:
            doc_names = self.dataset.get_required_documents(samples)
            if hasattr(self.dataset, "get_pdf_path"):
                pdf_paths = [self.dataset.get_pdf_path(d) for d in doc_names]
            else:
                pdf_paths = [self.data_dir / "pdfs" / f"{d}.pdf" for d in doc_names]

        doc_name_to_path: Dict[str, Path] = {}
        for p in pdf_paths:
            norm_name = CorpusSplitter._normalize_doc_name(p.name)
            doc_name_to_path[norm_name] = p

        # 3. Split Corpus and Queries
        corpus_split: CorpusSplit = CorpusSplitter.split(
            samples=samples,
            base_fraction=self.base_fraction,
            rounds=self.rounds,
        )

        print(f"[*] Corpus Split Summary:")
        print(f"    - Base Documents:   {corpus_split.base_doc_count} ({', '.join(corpus_split.base_docs)})")
        print(f"    - Update Documents: {corpus_split.update_doc_count} ({', '.join(corpus_split.update_docs) if corpus_split.update_docs else 'None'})")
        print(f"    - Base Queries:     {corpus_split.base_query_count}")
        print(f"    - New Queries:      {corpus_split.new_query_count}")
        print(f"    - Round Batches:    {len(corpus_split.round_partitions)}")

        # 4. Isolated Storage Setup
        if self.temp_dir:
            run_dir = self.temp_dir / f"run_{run_timestamp}"
        else:
            run_dir = self.output_dir / ".runs" / f"run_{run_timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        mode_a_dir = run_dir / "mode_a"
        mode_b_dir = run_dir / "mode_b"
        mode_a_dir.mkdir(parents=True, exist_ok=True)
        mode_b_dir.mkdir(parents=True, exist_ok=True)

        mode_a_chroma = str(mode_a_dir / "chroma")
        mode_a_manifest = str(mode_a_dir / "manifest.db")
        mode_b_chroma = str(mode_b_dir / "chroma")
        mode_b_manifest = str(mode_b_dir / "manifest.db")

        # -------------------------------------------------------------
        # 5. Mode A: Incremental Indexing & Multi-Round Execution
        # -------------------------------------------------------------
        print(f"\n--- [MODE A: INCREMENTAL INDEXING] ---")
        mode_a_indexer = BenchmarkIndexer(
            manifest_db_path=mode_a_manifest,
            chroma_path=mode_a_chroma,
            chroma_collection_name="mode_a_incremental",
        )

        # Base Phase (Round 0)
        base_pdfs = [
            doc_name_to_path[d]
            for d in corpus_split.base_docs
            if d in doc_name_to_path and doc_name_to_path[d].exists()
        ]
        print(f"[*] Ingesting Base Corpus ({len(base_pdfs)} documents)...")
        base_idx_stats = mode_a_indexer.index_documents(base_pdfs, show_progress=show_progress)
        print(
            f"[+] Base Corpus Indexed: {base_idx_stats['total_chunks']} chunks in "
            f"{base_idx_stats['total_indexing_time_seconds']}s ({base_idx_stats['total_embedding_calls']} embedding calls)."
        )

        # Mode A Pipeline for intermediate round evaluations
        mode_a_chroma_client = ChromaDBClient(
            path=mode_a_chroma,
            collection_name="mode_a_incremental",
        )
        mode_a_retriever = RetrievalService(chroma_client=mode_a_chroma_client)
        mode_a_rag = RAGService(retriever=mode_a_retriever, groq_service=self.groq_service)

        round_stress_test_stats: List[Dict[str, Any]] = []

        # Evaluate Round 0 baseline if multi-round
        baseline_base_hit5 = 0.0
        baseline_base_acc = 0.0
        if corpus_split.base_queries:
            r0_base_evals = self._evaluate_queries(
                corpus_split.base_queries,
                mode_a_retriever,
                mode_a_rag,
                desc="Evaluating Round 0 Base Queries",
                show_progress=show_progress,
            )
            baseline_base_hit5 = (
                sum(1 for r in r0_base_evals if r.retrieval_score.hit_at_5) / len(r0_base_evals)
                if r0_base_evals
                else 0.0
            )
            baseline_base_acc = (
                sum(1 for r in r0_base_evals if r.answer_score.is_correct) / len(r0_base_evals)
                if r0_base_evals
                else 0.0
            )

        round_stress_test_stats.append({
            "round_index": 0,
            "batch_docs_count": len(base_pdfs),
            "batch_chunks_count": base_idx_stats["total_chunks"],
            "embedding_calls": base_idx_stats["total_embedding_calls"],
            "round_time_seconds": base_idx_stats["total_indexing_time_seconds"],
            "base_query_hit_5": round(baseline_base_hit5, 4),
            "base_query_accuracy": round(baseline_base_acc, 4),
            "new_query_hit_5": 0.0,
            "new_query_accuracy": 0.0,
            "drift_from_round_0": 0.0,
        })

        # Incremental Update Phase(s)
        total_inc_update_time = 0.0
        total_inc_update_calls = 0
        total_inc_saved_calls = 0
        total_inc_chunks_added = 0

        # Partitions for rounds 1..R
        update_partitions = [p for p in corpus_split.round_partitions if p.round_index > 0]

        for p in update_partitions:
            round_pdfs = [
                doc_name_to_path[d]
                for d in p.batch_docs
                if d in doc_name_to_path and doc_name_to_path[d].exists()
            ]
            print(f"[*] Ingesting Incremental Round {p.round_index} ({len(round_pdfs)} documents)...")
            r_stats = mode_a_indexer.index_documents(round_pdfs, show_progress=show_progress)

            total_inc_update_time += r_stats["total_indexing_time_seconds"]
            total_inc_update_calls += r_stats["total_embedding_calls"]
            total_inc_saved_calls += r_stats["total_saved_calls"]
            total_inc_chunks_added += r_stats["total_chunks"]

            print(
                f"[+] Round {p.round_index} Indexed: {r_stats['total_chunks']} chunks in "
                f"{r_stats['total_indexing_time_seconds']}s ({r_stats['total_embedding_calls']} embedding calls, "
                f"{r_stats['total_saved_calls']} saved)."
            )

            # Re-evaluate Base queries to measure drift
            r_base_hit5 = 0.0
            r_base_acc = 0.0
            if corpus_split.base_queries:
                r_base_evals = self._evaluate_queries(
                    corpus_split.base_queries,
                    mode_a_retriever,
                    mode_a_rag,
                    desc=f"Evaluating Round {p.round_index} Base Queries",
                    show_progress=show_progress,
                )
                r_base_hit5 = sum(1 for r in r_base_evals if r.retrieval_score.hit_at_5) / len(r_base_evals)
                r_base_acc = sum(1 for r in r_base_evals if r.answer_score.is_correct) / len(r_base_evals)

            # Evaluate round's new queries
            r_new_hit5 = 0.0
            r_new_acc = 0.0
            if p.new_queries:
                r_new_evals = self._evaluate_queries(
                    p.new_queries,
                    mode_a_retriever,
                    mode_a_rag,
                    desc=f"Evaluating Round {p.round_index} New Queries",
                    show_progress=show_progress,
                )
                r_new_hit5 = sum(1 for r in r_new_evals if r.retrieval_score.hit_at_5) / len(r_new_evals)
                r_new_acc = sum(1 for r in r_new_evals if r.answer_score.is_correct) / len(r_new_evals)

            round_stress_test_stats.append({
                "round_index": p.round_index,
                "batch_docs_count": len(round_pdfs),
                "batch_chunks_count": r_stats["total_chunks"],
                "embedding_calls": r_stats["total_embedding_calls"],
                "round_time_seconds": r_stats["total_indexing_time_seconds"],
                "base_query_hit_5": round(r_base_hit5, 4),
                "base_query_accuracy": round(r_base_acc, 4),
                "new_query_hit_5": round(r_new_hit5, 4),
                "new_query_accuracy": round(r_new_acc, 4),
                "drift_from_round_0": round(r_base_acc - baseline_base_acc, 4),
            })

        # Final Mode A Evaluation on all Base and New queries
        print(f"[*] Running Final Evaluation on Mode A (Incremental)...")
        mode_a_base_results = self._evaluate_queries(
            corpus_split.base_queries,
            mode_a_retriever,
            mode_a_rag,
            desc="Mode A: Base Queries",
            show_progress=show_progress,
        )
        mode_a_new_results = self._evaluate_queries(
            corpus_split.new_queries,
            mode_a_retriever,
            mode_a_rag,
            desc="Mode A: New Queries",
            show_progress=show_progress,
        )

        # -------------------------------------------------------------
        # 6. Mode B: Full Rebuild Baseline
        # -------------------------------------------------------------
        print(f"\n--- [MODE B: FULL REBUILD BASELINE] ---")
        mode_b_indexer = BenchmarkIndexer(
            manifest_db_path=mode_b_manifest,
            chroma_path=mode_b_chroma,
            chroma_collection_name="mode_b_full_rebuild",
        )

        all_pdfs = [
            doc_name_to_path[d]
            for d in corpus_split.all_docs
            if d in doc_name_to_path and doc_name_to_path[d].exists()
        ]
        print(f"[*] Ingesting All Documents via Full Rebuild ({len(all_pdfs)} documents)...")
        full_rebuild_stats = mode_b_indexer.index_documents(all_pdfs, show_progress=show_progress)
        print(
            f"[+] Full Rebuild Complete: {full_rebuild_stats['total_chunks']} chunks in "
            f"{full_rebuild_stats['total_indexing_time_seconds']}s ({full_rebuild_stats['total_embedding_calls']} embedding calls)."
        )

        mode_b_chroma_client = ChromaDBClient(
            path=mode_b_chroma,
            collection_name="mode_b_full_rebuild",
        )
        mode_b_retriever = RetrievalService(chroma_client=mode_b_chroma_client)
        mode_b_rag = RAGService(retriever=mode_b_retriever, groq_service=self.groq_service)

        print(f"[*] Running Final Evaluation on Mode B (Full Rebuild)...")
        mode_b_base_results = self._evaluate_queries(
            corpus_split.base_queries,
            mode_b_retriever,
            mode_b_rag,
            desc="Mode B: Base Queries",
            show_progress=show_progress,
        )
        mode_b_new_results = self._evaluate_queries(
            corpus_split.new_queries,
            mode_b_retriever,
            mode_b_rag,
            desc="Mode B: New Queries",
            show_progress=show_progress,
        )

        # -------------------------------------------------------------
        # 7. Aggregate Metrics & Reporting
        # -------------------------------------------------------------
        corpus_stats = {
            "total_docs": len(corpus_split.all_docs),
            "base_doc_count": corpus_split.base_doc_count,
            "update_doc_count": corpus_split.update_doc_count,
            "base_fraction": self.base_fraction,
            "rounds": self.rounds,
            "total_queries": len(samples),
            "base_query_count": corpus_split.base_query_count,
            "new_query_count": corpus_split.new_query_count,
            "base_docs": corpus_split.base_docs,
            "update_docs": corpus_split.update_docs,
        }

        indexing_comparison = {
            "base_indexing_time_seconds": base_idx_stats["total_indexing_time_seconds"],
            "base_indexing_chunks": base_idx_stats["total_chunks"],
            "base_indexing_embedding_calls": base_idx_stats["total_embedding_calls"],
            "incremental_update_time_seconds": round(total_inc_update_time, 3),
            "incremental_update_chunks": total_inc_chunks_added,
            "incremental_update_embedding_calls": total_inc_update_calls,
            "incremental_update_saved_calls": total_inc_saved_calls,
            "full_rebuild_time_seconds": full_rebuild_stats["total_indexing_time_seconds"],
            "full_rebuild_chunks": full_rebuild_stats["total_chunks"],
            "full_rebuild_embedding_calls": full_rebuild_stats["total_embedding_calls"],
            "total_corpus_chunks": full_rebuild_stats["total_chunks"],
        }

        config = {
            "dataset": self.dataset_name,
            "base_fraction": self.base_fraction,
            "rounds": self.rounds,
            "top_k": self.top_k,
            "judge_mode": self.judge_mode,
        }

        reporter = IncrementalBenchmarkReporter(
            dataset_name=self.dataset_name,
            corpus_stats=corpus_stats,
            indexing_comparison=indexing_comparison,
            mode_a_base_results=mode_a_base_results,
            mode_a_new_results=mode_a_new_results,
            mode_b_base_results=mode_b_base_results,
            mode_b_new_results=mode_b_new_results,
            round_stress_test_stats=round_stress_test_stats,
            config=config,
        )

        json_file = reporter.save_json(output_dir=str(self.output_dir))
        md_file = reporter.save_markdown(output_dir=str(self.output_dir))

        print(f"\n[REPORT] Incremental benchmark results saved:")
        print(f"  - JSON Report:     {json_file}")
        print(f"  - Markdown Report: {md_file}")

        reporter.print_summary()

        # Cleanup isolated temp directory if requested
        if self.cleanup_temp and run_dir.exists():
            try:
                shutil.rmtree(run_dir, ignore_errors=True)
            except Exception:
                pass

        return reporter
