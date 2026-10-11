import time
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm

from app.eval.base import BaseBenchmarkDataset, SampleEvaluationResult
from app.eval.financebench_loader import FinanceBenchDataset
from app.eval.indexer import BenchmarkIndexer
from app.eval.metrics import RetrievalEvaluator, AnswerEvaluator
from app.eval.reporter import BenchmarkReporter

from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService
from app.services.groq_service import GroqService
from app.services.embedding_service import EmbeddingService
from app.database.chroma_db import ChromaDBClient


DATASET_REGISTRY = {
    "financebench": FinanceBenchDataset,
}


class EvaluationRunner:
    """Orchestrates benchmark evaluation runs over financial QA datasets."""

    def __init__(
        self,
        dataset_name: str = "financebench",
        data_dir: str = "data/financebench",
        output_dir: str = "benchmark_runs",
        manifest_db_path: str = "manifest_db/manifest.db",
        chroma_path: str = "chroma_storage",
        chroma_collection_name: str = "financebench_docs",
        top_k: int = 5,
        judge_mode: str = "hybrid",  # 'hybrid', 'semantic', 'llm', 'numeric'
        groq_service: Optional[GroqService] = None,
    ):
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.manifest_db_path = manifest_db_path
        self.chroma_path = chroma_path
        self.chroma_collection_name = chroma_collection_name
        self.top_k = top_k
        self.judge_mode = judge_mode

        # Initialize dataset loader
        dataset_cls = DATASET_REGISTRY.get(self.dataset_name)
        if not dataset_cls:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}"
            )
        self.dataset: BaseBenchmarkDataset = dataset_cls(data_dir=self.data_dir)

        # Indexer
        self.indexer = BenchmarkIndexer(
            manifest_db_path=self.manifest_db_path,
            chroma_path=self.chroma_path,
            chroma_collection_name=self.chroma_collection_name,
        )

        # Embedding & Groq services
        self.embedder = EmbeddingService()
        
        if groq_service is not None:
            self.groq_service = groq_service
        elif self.judge_mode in ("llm", "hybrid"):
            try:
                self.groq_service = GroqService()
            except Exception as e:
                print(f"Warning: GroqService could not be initialized ({e}). Defaulting judge_mode to 'semantic'.")
                self.groq_service = None
                if self.judge_mode in ("llm", "hybrid"):
                    self.judge_mode = "semantic"
        else:
            self.groq_service = None

        # Evaluator components
        self.answer_evaluator = AnswerEvaluator(
            embedder=self.embedder,
            groq_service=self.groq_service,
        )

    def run(
        self,
        n: Optional[int] = None,
        skip_ingestion: bool = False,
        show_progress: bool = True,
    ) -> BenchmarkReporter:
        """Run the full benchmark evaluation workflow."""
        print(f"\n[*] Starting benchmark evaluation on '{self.dataset_name}' (n={n or 'all'})...")

        # 1. Load questions
        samples = self.dataset.load(n=n)
        print(f"[*] Loaded {len(samples)} questions from {self.dataset_name}.")

        # 2. Ingestion
        indexing_stats = {}
        if not skip_ingestion and isinstance(self.dataset, FinanceBenchDataset):
            pdf_paths = self.dataset.download_pdfs_for_samples(samples)
            print(f"[*] Ensuring {len(pdf_paths)} documents are indexed into ChromaDB ({self.chroma_collection_name})...")
            indexing_stats = self.indexer.index_documents(pdf_paths, show_progress=show_progress)
            print(f"[+] Indexing completed: {indexing_stats['total_chunks']} chunks ({indexing_stats['total_embedding_calls']} new embeddings generated).")

        # 3. Setup RAG pipeline
        chroma_client = ChromaDBClient(
            path=self.chroma_path,
            collection_name=self.chroma_collection_name,
        )
        retriever = RetrievalService(chroma_client=chroma_client)
        rag_service = RAGService(
            retriever=retriever,
            groq_service=self.groq_service,
        )

        # 4. Evaluation Loop
        results: List[SampleEvaluationResult] = []
        overall_start = time.perf_counter()

        iterator = tqdm(samples, desc="Evaluating Questions") if show_progress else samples

        for sample in iterator:
            q_start = time.perf_counter()

            if self.groq_service:
                rag_out = rag_service.ask(sample.question, top_k=self.top_k, log_analytics=False)
                generated_answer = rag_out.get("answer", "")
                retrieved_chunks = rag_out.get("sources", [])
            else:
                # If no Groq API key, evaluate retrieval directly
                retrieved_chunks = retriever.retrieve(sample.question, top_k=self.top_k)
                generated_answer = "Groq LLM disabled / unavailable"

            q_latency = round(time.perf_counter() - q_start, 3)

            # Retrieval scoring
            ret_score = RetrievalEvaluator.evaluate_retrieval(
                retrieved_chunks=retrieved_chunks,
                sample=sample,
                top_k=self.top_k,
            )

            # Answer quality scoring
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

        total_time = round(time.perf_counter() - overall_start, 2)

        # 5. Reporting
        config = {
            "dataset": self.dataset_name,
            "top_k": self.top_k,
            "judge_mode": self.judge_mode,
            "collection_name": self.chroma_collection_name,
            "chroma_path": self.chroma_path,
        }

        reporter = BenchmarkReporter(
            dataset_name=self.dataset_name,
            results=results,
            total_time_seconds=total_time,
            indexing_stats=indexing_stats,
            config=config,
        )

        # Save artifacts
        json_file = reporter.save_json(output_dir=self.output_dir)
        md_file = reporter.save_markdown(output_dir=self.output_dir)

        print(f"\n[REPORT] Benchmark results saved:")
        print(f"  - JSON Report:     {json_file}")
        print(f"  - Markdown Report: {md_file}")

        reporter.print_summary()
        return reporter
