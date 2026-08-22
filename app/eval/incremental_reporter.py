import os
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.eval.base import SampleEvaluationResult


class IncrementalBenchmarkReporter:
    """
    Aggregates, formats, and saves comparative reports for Incremental RAG vs. Full Rebuild RAG.
    Produces:
    1. Comprehensive JSON report artifact in benchmark_runs/
    2. Human-readable Markdown summary in benchmark_runs/
    3. Formatted terminal summary tables.
    """

    def __init__(
        self,
        dataset_name: str,
        corpus_stats: Dict[str, Any],
        indexing_comparison: Dict[str, Any],
        mode_a_base_results: List[SampleEvaluationResult],
        mode_a_new_results: List[SampleEvaluationResult],
        mode_b_base_results: List[SampleEvaluationResult],
        mode_b_new_results: List[SampleEvaluationResult],
        round_stress_test_stats: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_name = dataset_name
        self.corpus_stats = corpus_stats
        self.indexing = indexing_comparison
        self.mode_a_base = mode_a_base_results
        self.mode_a_new = mode_a_new_results
        self.mode_b_base = mode_b_base_results
        self.mode_b_new = mode_b_new_results
        self.round_stress_test_stats = round_stress_test_stats or []
        self.config = config or {}
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.summary = self._compute_summary()

    def _compute_metrics(self, results: List[SampleEvaluationResult]) -> Dict[str, Any]:
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "document_hit_rate": 0.0,
                "page_hit_rate": 0.0,
                "hit_at_1": 0.0,
                "hit_at_3": 0.0,
                "hit_at_5": 0.0,
                "mrr": 0.0,
                "average_text_overlap": 0.0,
                "accuracy": 0.0,
                "exact_match_rate": 0.0,
                "numeric_match_rate": 0.0,
                "average_semantic_similarity": 0.0,
                "llm_judge_accuracy": None,
                "average_latency_seconds": 0.0,
            }

        doc_hits = sum(1 for r in results if r.retrieval_score.doc_hit)
        page_hits = sum(1 for r in results if r.retrieval_score.page_hit)
        hit_1 = sum(1 for r in results if r.retrieval_score.hit_at_1)
        hit_3 = sum(1 for r in results if r.retrieval_score.hit_at_3)
        hit_5 = sum(1 for r in results if r.retrieval_score.hit_at_5)
        mean_mrr = sum(r.retrieval_score.mrr for r in results) / total
        avg_text_overlap = sum(r.retrieval_score.text_overlap_score for r in results) / total

        correct_answers = sum(1 for r in results if r.answer_score.is_correct)
        exact_matches = sum(1 for r in results if r.answer_score.exact_match)
        numeric_matches = sum(1 for r in results if r.answer_score.numeric_match)
        avg_semantic_sim = sum(r.answer_score.semantic_similarity for r in results) / total

        llm_judged = [r for r in results if r.answer_score.llm_judge_score is not None]
        llm_acc = (
            sum(1 for r in llm_judged if r.answer_score.is_correct) / len(llm_judged)
            if llm_judged
            else None
        )

        avg_latency = sum(r.latency_seconds for r in results) / total

        return {
            "total": total,
            "document_hit_rate": round(doc_hits / total, 4),
            "page_hit_rate": round(page_hits / total, 4),
            "hit_at_1": round(hit_1 / total, 4),
            "hit_at_3": round(hit_3 / total, 4),
            "hit_at_5": round(hit_5 / total, 4),
            "mrr": round(mean_mrr, 4),
            "average_text_overlap": round(avg_text_overlap, 4),
            "accuracy": round(correct_answers / total, 4),
            "exact_match_rate": round(exact_matches / total, 4),
            "numeric_match_rate": round(numeric_matches / total, 4),
            "average_semantic_similarity": round(avg_semantic_sim, 4),
            "llm_judge_accuracy": round(llm_acc, 4) if llm_acc is not None else None,
            "average_latency_seconds": round(avg_latency, 3),
        }

    def _compute_delta(self, mode_a_m: Dict[str, Any], mode_b_m: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Mode A - Mode B parity deltas."""
        return {
            "page_hit_rate_delta": round(mode_a_m["page_hit_rate"] - mode_b_m["page_hit_rate"], 4),
            "hit_at_5_delta": round(mode_a_m["hit_at_5"] - mode_b_m["hit_at_5"], 4),
            "mrr_delta": round(mode_a_m["mrr"] - mode_b_m["mrr"], 4),
            "accuracy_delta": round(mode_a_m["accuracy"] - mode_b_m["accuracy"], 4),
            "semantic_similarity_delta": round(
                mode_a_m["average_semantic_similarity"] - mode_b_m["average_semantic_similarity"], 4
            ),
        }

    def _compute_summary(self) -> Dict[str, Any]:
        # Aggregate Mode A and Mode B all results
        mode_a_all = self.mode_a_base + self.mode_a_new
        mode_b_all = self.mode_b_base + self.mode_b_new

        a_base_m = self._compute_metrics(self.mode_a_base)
        a_new_m = self._compute_metrics(self.mode_a_new)
        a_all_m = self._compute_metrics(mode_a_all)

        b_base_m = self._compute_metrics(self.mode_b_base)
        b_new_m = self._compute_metrics(self.mode_b_new)
        b_all_m = self._compute_metrics(mode_b_all)

        base_delta = self._compute_delta(a_base_m, b_base_m)
        new_delta = self._compute_delta(a_new_m, b_new_m)
        all_delta = self._compute_delta(a_all_m, b_all_m)

        # Efficiency calculations
        inc_update_time = self.indexing.get("incremental_update_time_seconds", 0.0)
        full_rebuild_time = self.indexing.get("full_rebuild_time_seconds", 0.0)
        inc_embedding_calls = self.indexing.get("incremental_update_embedding_calls", 0)
        full_embedding_calls = self.indexing.get("full_rebuild_embedding_calls", 0)
        total_corpus_chunks = self.indexing.get("total_corpus_chunks", max(1, full_embedding_calls))

        time_saved = max(0.0, full_rebuild_time - inc_update_time)
        time_saved_pct = (time_saved / full_rebuild_time * 100.0) if full_rebuild_time > 0 else 0.0
        calls_saved = max(0, full_embedding_calls - inc_embedding_calls)
        calls_saved_pct = (calls_saved / full_embedding_calls * 100.0) if full_embedding_calls > 0 else 0.0
        reembed_ratio = (inc_embedding_calls / total_corpus_chunks) if total_corpus_chunks > 0 else 0.0
        speedup = (full_rebuild_time / inc_update_time) if inc_update_time > 0 else 1.0

        return {
            "dataset": self.dataset_name,
            "timestamp": self.timestamp,
            "corpus": self.corpus_stats,
            "efficiency": {
                "incremental_update_time_seconds": round(inc_update_time, 3),
                "full_rebuild_time_seconds": round(full_rebuild_time, 3),
                "time_saved_seconds": round(time_saved, 3),
                "time_saved_percentage": round(time_saved_pct, 2),
                "speedup_factor": round(speedup, 2),
                "incremental_update_embedding_calls": inc_embedding_calls,
                "full_rebuild_embedding_calls": full_embedding_calls,
                "embedding_calls_saved": calls_saved,
                "embedding_calls_saved_percentage": round(calls_saved_pct, 2),
                "chunk_reembedding_ratio": round(reembed_ratio, 4),
                "total_corpus_chunks": total_corpus_chunks,
            },
            "quality_parity": {
                "base_queries": {
                    "mode_a_incremental": a_base_m,
                    "mode_b_full_rebuild": b_base_m,
                    "delta": base_delta,
                },
                "new_queries": {
                    "mode_a_incremental": a_new_m,
                    "mode_b_full_rebuild": b_new_m,
                    "delta": new_delta,
                },
                "overall": {
                    "mode_a_incremental": a_all_m,
                    "mode_b_full_rebuild": b_all_m,
                    "delta": all_delta,
                },
            },
            "multi_round_stress_test": self.round_stress_test_stats,
            "config": self.config,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert full evaluation results to serializable dictionary."""
        # Create map of sample evaluations
        sample_details = []

        mode_a_dict = {r.sample.id: r for r in (self.mode_a_base + self.mode_a_new)}
        mode_b_dict = {r.sample.id: r for r in (self.mode_b_base + self.mode_b_new)}

        all_sample_ids = list(mode_a_dict.keys())

        for sid in all_sample_ids:
            ra = mode_a_dict.get(sid)
            rb = mode_b_dict.get(sid)
            if not ra:
                continue

            is_base = any(r.sample.id == sid for r in self.mode_a_base)

            sample_details.append({
                "id": ra.sample.id,
                "query_category": "base_query" if is_base else "new_query",
                "company": ra.sample.company,
                "doc_name": ra.sample.doc_name,
                "question": ra.sample.question,
                "gold_answer": ra.sample.answer,
                "mode_a_incremental": {
                    "generated_answer": ra.generated_answer,
                    "is_correct": ra.answer_score.is_correct,
                    "score": ra.answer_score.overall_score,
                    "page_hit": ra.retrieval_score.page_hit,
                    "doc_hit": ra.retrieval_score.doc_hit,
                    "hit_at_5": ra.retrieval_score.hit_at_5,
                    "mrr": ra.retrieval_score.mrr,
                    "retrieved_pages": ra.retrieval_score.retrieved_pages_1_indexed,
                },
                "mode_b_full_rebuild": {
                    "generated_answer": rb.generated_answer if rb else "",
                    "is_correct": rb.answer_score.is_correct if rb else False,
                    "score": rb.answer_score.overall_score if rb else 0.0,
                    "page_hit": rb.retrieval_score.page_hit if rb else False,
                    "doc_hit": rb.retrieval_score.doc_hit if rb else False,
                    "hit_at_5": rb.retrieval_score.hit_at_5 if rb else False,
                    "mrr": rb.retrieval_score.mrr if rb else 0.0,
                    "retrieved_pages": rb.retrieval_score.retrieved_pages_1_indexed if rb else [],
                },
            })

        return {
            "summary": self.summary,
            "indexing_breakdown": self.indexing,
            "samples": sample_details,
        }

    def save_json(self, output_dir: str = "benchmark_runs") -> Path:
        """Save incremental evaluation details to a JSON file."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"incremental_eval_{self.timestamp}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        return file_path

    def save_markdown(self, output_dir: str = "benchmark_runs") -> Path:
        """Save human-readable Markdown summary report with comparative tables."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"incremental_summary_{self.timestamp}.md"

        eff = self.summary["efficiency"]
        qp = self.summary["quality_parity"]
        corp = self.summary["corpus"]

        md_content = f"""# Incremental RAG Efficiency & Quality Benchmark Report

- **Date / Timestamp**: `{self.timestamp}`
- **Dataset**: `{self.dataset_name.upper()}`
- **Corpus Split**: `{corp.get('base_doc_count', 0)}` Base Docs ({corp.get('base_fraction', 0.8)*100:.0f}%) / `{corp.get('update_doc_count', 0)}` Update Docs
- **Evaluated Queries**: `{corp.get('base_query_count', 0)}` Base Queries + `{corp.get('new_query_count', 0)}` New Queries (`{corp.get('total_queries', 0)}` Total)
- **Incremental Rounds**: `{corp.get('rounds', 1)}` sequential batch(es)

---

## 1. Executive Summary: Incremental Efficiency & Parity

| Metric | Mode A (Incremental Update) | Mode B (Full Rebuild) | Efficiency Gain / Parity Delta |
| :--- | :--- | :--- | :--- |
| **Index Update Time** | **`{eff['incremental_update_time_seconds']}s`** | `{eff['full_rebuild_time_seconds']}s` | **`{eff['speedup_factor']}x` Faster** (`{eff['time_saved_percentage']:.1f}%` saved) |
| **Embedding Calls** | **`{eff['incremental_update_embedding_calls']}`** | `{eff['full_rebuild_embedding_calls']}` | **`{eff['embedding_calls_saved_percentage']:.1f}%` Saved** (`{eff['embedding_calls_saved']}` calls skipped) |
| **Re-embedded Chunk Ratio** | **`{eff['chunk_reembedding_ratio']*100:.1f}%`** | `100.0%` | **Only changed/new chunks processed** |
| **Base Query Hit@5** | `{qp['base_queries']['mode_a_incremental']['hit_at_5']*100:.1f}%` | `{qp['base_queries']['mode_b_full_rebuild']['hit_at_5']*100:.1f}%` | `Δ {qp['base_queries']['delta']['hit_at_5_delta']*100:+.1f}%` |
| **New Query Hit@5** | `{qp['new_queries']['mode_a_incremental']['hit_at_5']*100:.1f}%` | `{qp['new_queries']['mode_b_full_rebuild']['hit_at_5']*100:.1f}%` | `Δ {qp['new_queries']['delta']['hit_at_5_delta']*100:+.1f}%` |
| **Overall Answer Accuracy** | **`{qp['overall']['mode_a_incremental']['accuracy']*100:.1f}%`** | **`{qp['overall']['mode_b_full_rebuild']['accuracy']*100:.1f}%`** | **`Δ {qp['overall']['delta']['accuracy_delta']*100:+.1f}%`** |

---

## 2. Retrieval Parity Breakdown (Mode A vs Mode B)

*Verifies that incremental indexing achieves the exact same retrieval quality as rebuilding from scratch (ECT-QA & EraRAG methodology).*

| Query Group | Metric | Mode A (Incremental) | Mode B (Full Rebuild) | Delta (A - B) | Parity Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Queries** (`n={corp.get('base_query_count', 0)}`) | **Hit@1** | `{qp['base_queries']['mode_a_incremental']['hit_at_1']*100:.1f}%` | `{qp['base_queries']['mode_b_full_rebuild']['hit_at_1']*100:.1f}%` | `{qp['base_queries']['mode_a_incremental']['hit_at_1'] - qp['base_queries']['mode_b_full_rebuild']['hit_at_1']:+.3f}` | {"✅ Parity" if abs(qp['base_queries']['mode_a_incremental']['hit_at_1'] - qp['base_queries']['mode_b_full_rebuild']['hit_at_1']) < 0.05 else "⚠️ Slight Delta"} |
| | **Hit@5** | `{qp['base_queries']['mode_a_incremental']['hit_at_5']*100:.1f}%` | `{qp['base_queries']['mode_b_full_rebuild']['hit_at_5']*100:.1f}%` | `{qp['base_queries']['delta']['hit_at_5_delta']:+.3f}` | {"✅ Parity" if abs(qp['base_queries']['delta']['hit_at_5_delta']) < 0.05 else "⚠️ Slight Delta"} |
| | **Page Hit Rate** | `{qp['base_queries']['mode_a_incremental']['page_hit_rate']*100:.1f}%` | `{qp['base_queries']['mode_b_full_rebuild']['page_hit_rate']*100:.1f}%` | `{qp['base_queries']['delta']['page_hit_rate_delta']:+.3f}` | {"✅ Parity" if abs(qp['base_queries']['delta']['page_hit_rate_delta']) < 0.05 else "⚠️ Slight Delta"} |
| | **MRR** | `{qp['base_queries']['mode_a_incremental']['mrr']:.4f}` | `{qp['base_queries']['mode_b_full_rebuild']['mrr']:.4f}` | `{qp['base_queries']['delta']['mrr_delta']:+.4f}` | {"✅ Parity" if abs(qp['base_queries']['delta']['mrr_delta']) < 0.05 else "⚠️ Slight Delta"} |
| **New Queries** (`n={corp.get('new_query_count', 0)}`) | **Hit@1** | `{qp['new_queries']['mode_a_incremental']['hit_at_1']*100:.1f}%` | `{qp['new_queries']['mode_b_full_rebuild']['hit_at_1']*100:.1f}%` | `{qp['new_queries']['mode_a_incremental']['hit_at_1'] - qp['new_queries']['mode_b_full_rebuild']['hit_at_1']:+.3f}` | {"✅ Parity" if abs(qp['new_queries']['mode_a_incremental']['hit_at_1'] - qp['new_queries']['mode_b_full_rebuild']['hit_at_1']) < 0.05 else "⚠️ Slight Delta"} |
| | **Hit@5** | `{qp['new_queries']['mode_a_incremental']['hit_at_5']*100:.1f}%` | `{qp['new_queries']['mode_b_full_rebuild']['hit_at_5']*100:.1f}%` | `{qp['new_queries']['delta']['hit_at_5_delta']:+.3f}` | {"✅ Parity" if abs(qp['new_queries']['delta']['hit_at_5_delta']) < 0.05 else "⚠️ Slight Delta"} |
| | **Page Hit Rate** | `{qp['new_queries']['mode_a_incremental']['page_hit_rate']*100:.1f}%` | `{qp['new_queries']['mode_b_full_rebuild']['page_hit_rate']*100:.1f}%` | `{qp['new_queries']['delta']['page_hit_rate_delta']:+.3f}` | {"✅ Parity" if abs(qp['new_queries']['delta']['page_hit_rate_delta']) < 0.05 else "⚠️ Slight Delta"} |
| | **MRR** | `{qp['new_queries']['mode_a_incremental']['mrr']:.4f}` | `{qp['new_queries']['mode_b_full_rebuild']['mrr']:.4f}` | `{qp['new_queries']['delta']['mrr_delta']:+.4f}` | {"✅ Parity" if abs(qp['new_queries']['delta']['mrr_delta']) < 0.05 else "⚠️ Slight Delta"} |
| **All Queries** (`n={corp.get('total_queries', 0)}`) | **Hit@5** | **`{qp['overall']['mode_a_incremental']['hit_at_5']*100:.1f}%`** | **`{qp['overall']['mode_b_full_rebuild']['hit_at_5']*100:.1f}%`** | **`{qp['overall']['delta']['hit_at_5_delta']:+.3f}`** | **{"✅ Parity" if abs(qp['overall']['delta']['hit_at_5_delta']) < 0.05 else "⚠️ Slight Delta"}** |
| | **Page Hit Rate** | **`{qp['overall']['mode_a_incremental']['page_hit_rate']*100:.1f}%`** | **`{qp['overall']['mode_b_full_rebuild']['page_hit_rate']*100:.1f}%`** | **`{qp['overall']['delta']['page_hit_rate_delta']:+.3f}`** | **{"✅ Parity" if abs(qp['overall']['delta']['page_hit_rate_delta']) < 0.05 else "⚠️ Slight Delta"}** |
| | **Mean MRR** | **`{qp['overall']['mode_a_incremental']['mrr']:.4f}`** | **`{qp['overall']['mode_b_full_rebuild']['mrr']:.4f}`** | **`{qp['overall']['delta']['mrr_delta']:+.4f}`** | **{"✅ Parity" if abs(qp['overall']['delta']['mrr_delta']) < 0.05 else "⚠️ Slight Delta"}** |

---

## 3. Answer Quality Parity Breakdown

| Query Group | Metric | Mode A (Incremental) | Mode B (Full Rebuild) | Delta (A - B) |
| :--- | :--- | :--- | :--- | :--- |
| **Base Queries** | Overall Accuracy | `{qp['base_queries']['mode_a_incremental']['accuracy']*100:.1f}%` | `{qp['base_queries']['mode_b_full_rebuild']['accuracy']*100:.1f}%` | `{qp['base_queries']['delta']['accuracy_delta']:+.3f}` |
| | Semantic Similarity | `{qp['base_queries']['mode_a_incremental']['average_semantic_similarity']*100:.1f}%` | `{qp['base_queries']['mode_b_full_rebuild']['average_semantic_similarity']*100:.1f}%` | `{qp['base_queries']['delta']['semantic_similarity_delta']:+.3f}` |
| **New Queries** | Overall Accuracy | `{qp['new_queries']['mode_a_incremental']['accuracy']*100:.1f}%` | `{qp['new_queries']['mode_b_full_rebuild']['accuracy']*100:.1f}%` | `{qp['new_queries']['delta']['accuracy_delta']:+.3f}` |
| | Semantic Similarity | `{qp['new_queries']['mode_a_incremental']['average_semantic_similarity']*100:.1f}%` | `{qp['new_queries']['mode_b_full_rebuild']['average_semantic_similarity']*100:.1f}%` | `{qp['new_queries']['delta']['semantic_similarity_delta']:+.3f}` |
| **All Queries** | **Overall Accuracy** | **`{qp['overall']['mode_a_incremental']['accuracy']*100:.1f}%`** | **`{qp['overall']['mode_b_full_rebuild']['accuracy']*100:.1f}%`** | **`{qp['overall']['delta']['accuracy_delta']:+.3f}`** |
| | **Avg Semantic Sim** | **`{qp['overall']['mode_a_incremental']['average_semantic_similarity']*100:.1f}%`** | **`{qp['overall']['mode_b_full_rebuild']['average_semantic_similarity']*100:.1f}%`** | **`{qp['overall']['delta']['semantic_similarity_delta']:+.3f}`** |

---
"""

        # Section 4: Multi-Round Stress Test (if available)
        if self.round_stress_test_stats:
            md_content += """## 4. Multi-Round Sequential Stress Test

*Tracks retrieval stability and drift on Base Queries as sequential document updates are indexed into the database.*

| Round | Stage | Docs Ingested | Chunks Added | Embed Calls | Round Time | Base Query Hit@5 | Base Query Acc. | New Query Hit@5 | Base Drift |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
            for r in self.round_stress_test_stats:
                r_idx = r.get("round_index", 0)
                stage = "Base Corpus" if r_idx == 0 else f"Update Batch {r_idx}"
                drift_str = f"{r.get('drift_from_round_0', 0.0)*100:+.1f}%" if r_idx > 0 else "0.0% (Baseline)"
                new_hit_str = f"{r.get('new_query_hit_5', 0.0)*100:.1f}%" if r_idx > 0 else "N/A"

                md_content += (
                    f"| `Round {r_idx}` | {stage} | `{r.get('batch_docs_count', 0)}` "
                    f"| `{r.get('batch_chunks_count', 0)}` | `{r.get('embedding_calls', 0)}` "
                    f"| `{r.get('round_time_seconds', 0.0):.2f}s` | `{r.get('base_query_hit_5', 0.0)*100:.1f}%` "
                    f"| `{r.get('base_query_accuracy', 0.0)*100:.1f}%` | `{new_hit_str}` "
                    f"| `{drift_str}` |\n"
                )

            md_content += "\n---\n"

        # Section 5: Sample breakdown
        md_content += """## 5. Sample Breakdown (First 15 Queries)

| ID | Type | Company | Question Snippet | Mode A Hit | Mode B Hit | Mode A Acc. | Mode B Acc. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        mode_a_map = {r.sample.id: r for r in (self.mode_a_base + self.mode_a_new)}
        mode_b_map = {r.sample.id: r for r in (self.mode_b_base + self.mode_b_new)}

        displayed_samples = (self.mode_a_base + self.mode_a_new)[:15]
        for ra in displayed_samples:
            rb = mode_b_map.get(ra.sample.id)
            is_base = any(r.sample.id == ra.sample.id for r in self.mode_a_base)
            type_str = "Base" if is_base else "New"
            q_snip = (ra.sample.question[:40] + "...") if len(ra.sample.question) > 40 else ra.sample.question
            q_snip = q_snip.replace("|", "/")

            hit_a = "✅" if ra.retrieval_score.hit_at_5 else "❌"
            hit_b = ("✅" if rb.retrieval_score.hit_at_5 else "❌") if rb else "N/A"
            acc_a = "✅" if ra.answer_score.is_correct else "❌"
            acc_b = ("✅" if rb.answer_score.is_correct else "❌") if rb else "N/A"

            md_content += f"| `{ra.sample.id}` | `{type_str}` | {ra.sample.company or 'N/A'} | {q_snip} | {hit_a} | {hit_b} | {acc_a} | {acc_b} |\n"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return file_path

    def print_summary(self):
        """Print an ASCII summary table to console."""
        eff = self.summary["efficiency"]
        qp = self.summary["quality_parity"]
        corp = self.summary["corpus"]

        print("\n" + "=" * 85)
        print(f"       INCREMENTAL RAG BENCHMARK SUMMARY: {self.dataset_name.upper()}")
        print("=" * 85)
        print(f"Corpus: {corp.get('base_doc_count', 0)} Base Docs + {corp.get('update_doc_count', 0)} Update Docs ({corp.get('total_docs', 0)} Total)")
        print(f"Queries: {corp.get('base_query_count', 0)} Base Queries + {corp.get('new_query_count', 0)} New Queries ({corp.get('total_queries', 0)} Total)")
        print("-" * 85)
        print(f"{'EFFICIENCY METRIC':<35} | {'MODE A (INCREMENTAL)':<22} | {'MODE B (FULL REBUILD)':<22}")
        print("-" * 85)
        print(f"{'Index Build / Update Time':<35} | {eff['incremental_update_time_seconds']:>19.2f}s | {eff['full_rebuild_time_seconds']:>19.2f}s")
        print(f"{'Time Saved':<35} | {eff['time_saved_percentage']:>18.1f}% | {'(Speedup: ' + str(eff['speedup_factor']) + 'x)':>22}")
        print(f"{'Embedding / Re-index Calls':<35} | {eff['incremental_update_embedding_calls']:>20} | {eff['full_rebuild_embedding_calls']:>22}")
        print(f"{'Embedding Calls Saved':<35} | {eff['embedding_calls_saved_percentage']:>18.1f}% | {eff['embedding_calls_saved']:>16} saved")
        print(f"{'Chunk Re-embedded Ratio':<35} | {eff['chunk_reembedding_ratio']*100:>18.1f}% | {'100.0%':>22}")
        print("-" * 85)
        print(f"{'QUALITY PARITY (HIT@5)':<35} | {'MODE A (INCREMENTAL)':<22} | {'MODE B (FULL REBUILD)':<22}")
        print("-" * 85)
        print(f"{'Base Queries Hit@5':<35} | {qp['base_queries']['mode_a_incremental']['hit_at_5']*100:>19.1f}% | {qp['base_queries']['mode_b_full_rebuild']['hit_at_5']*100:>19.1f}% (Δ {qp['base_queries']['delta']['hit_at_5_delta']*100:+.1f}%)")
        print(f"{'New Queries Hit@5':<35} | {qp['new_queries']['mode_a_incremental']['hit_at_5']*100:>19.1f}% | {qp['new_queries']['mode_b_full_rebuild']['hit_at_5']*100:>19.1f}% (Δ {qp['new_queries']['delta']['hit_at_5_delta']*100:+.1f}%)")
        print(f"{'Overall Hit@5':<35} | {qp['overall']['mode_a_incremental']['hit_at_5']*100:>19.1f}% | {qp['overall']['mode_b_full_rebuild']['hit_at_5']*100:>19.1f}% (Δ {qp['overall']['delta']['hit_at_5_delta']*100:+.1f}%)")
        print(f"{'Overall Answer Accuracy':<35} | {qp['overall']['mode_a_incremental']['accuracy']*100:>19.1f}% | {qp['overall']['mode_b_full_rebuild']['accuracy']*100:>19.1f}% (Δ {qp['overall']['delta']['accuracy_delta']*100:+.1f}%)")
        print(f"{'Mean Reciprocal Rank (MRR)':<35} | {qp['overall']['mode_a_incremental']['mrr']:>20.4f} | {qp['overall']['mode_b_full_rebuild']['mrr']:>22.4f}")
        print("=" * 85 + "\n")
