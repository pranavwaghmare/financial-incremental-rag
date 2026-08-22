import os
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.eval.base import SampleEvaluationResult


class BenchmarkReporter:
    """Aggregates benchmark results, saves JSON / Markdown reports, and formats terminal summaries."""

    def __init__(
        self,
        dataset_name: str,
        results: List[SampleEvaluationResult],
        total_time_seconds: float,
        indexing_stats: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_name = dataset_name
        self.results = results
        self.total_time_seconds = total_time_seconds
        self.indexing_stats = indexing_stats or {}
        self.config = config or {}
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary = self._compute_summary()

    def _compute_summary(self) -> Dict[str, Any]:
        total = len(self.results)
        if total == 0:
            return {"total_samples": 0}

        doc_hits = sum(1 for r in self.results if r.retrieval_score.doc_hit)
        page_hits = sum(1 for r in self.results if r.retrieval_score.page_hit)
        hit_1 = sum(1 for r in self.results if r.retrieval_score.hit_at_1)
        hit_3 = sum(1 for r in self.results if r.retrieval_score.hit_at_3)
        hit_5 = sum(1 for r in self.results if r.retrieval_score.hit_at_5)
        mean_mrr = sum(r.retrieval_score.mrr for r in self.results) / total
        avg_text_overlap = sum(r.retrieval_score.text_overlap_score for r in self.results) / total

        correct_answers = sum(1 for r in self.results if r.answer_score.is_correct)
        exact_matches = sum(1 for r in self.results if r.answer_score.exact_match)
        numeric_matches = sum(1 for r in self.results if r.answer_score.numeric_match)
        avg_semantic_sim = sum(r.answer_score.semantic_similarity for r in self.results) / total

        llm_judged = [r for r in self.results if r.answer_score.llm_judge_score is not None]
        llm_accuracy = (
            sum(1 for r in llm_judged if r.answer_score.is_correct) / len(llm_judged)
            if llm_judged
            else None
        )

        avg_latency = sum(r.latency_seconds for r in self.results) / total

        return {
            "dataset": self.dataset_name,
            "timestamp": self.timestamp,
            "total_samples": total,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "average_latency_seconds": round(avg_latency, 3),
            "retrieval": {
                "document_hit_rate": round(doc_hits / total, 4),
                "page_hit_rate": round(page_hits / total, 4),
                "hit_at_1": round(hit_1 / total, 4),
                "hit_at_3": round(hit_3 / total, 4),
                "hit_at_5": round(hit_5 / total, 4),
                "mrr": round(mean_mrr, 4),
                "average_text_overlap": round(avg_text_overlap, 4),
            },
            "answer_quality": {
                "accuracy": round(correct_answers / total, 4),
                "exact_match_rate": round(exact_matches / total, 4),
                "numeric_match_rate": round(numeric_matches / total, 4),
                "average_semantic_similarity": round(avg_semantic_sim, 4),
                "llm_judge_accuracy": round(llm_accuracy, 4) if llm_accuracy is not None else None,
            },
            "indexing": self.indexing_stats,
            "config": self.config,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert full evaluation results to serializable dictionary."""
        detailed_samples = []
        for r in self.results:
            detailed_samples.append({
                "id": r.sample.id,
                "company": r.sample.company,
                "doc_name": r.sample.doc_name,
                "question_type": r.sample.question_type,
                "question": r.sample.question,
                "gold_answer": r.sample.answer,
                "gold_justification": r.sample.justification,
                "generated_answer": r.generated_answer,
                "latency_seconds": round(r.latency_seconds, 3),
                "retrieval": {
                    "doc_hit": r.retrieval_score.doc_hit,
                    "page_hit": r.retrieval_score.page_hit,
                    "hit_at_1": r.retrieval_score.hit_at_1,
                    "hit_at_3": r.retrieval_score.hit_at_3,
                    "hit_at_5": r.retrieval_score.hit_at_5,
                    "mrr": r.retrieval_score.mrr,
                    "text_overlap_score": r.retrieval_score.text_overlap_score,
                    "gold_pages_1_indexed": r.retrieval_score.gold_pages_1_indexed,
                    "retrieved_pages_1_indexed": r.retrieval_score.retrieved_pages_1_indexed,
                    "retrieved_docs": r.retrieval_score.retrieved_docs,
                },
                "answer_quality": {
                    "is_correct": r.answer_score.is_correct,
                    "overall_score": r.answer_score.overall_score,
                    "exact_match": r.answer_score.exact_match,
                    "numeric_match": r.answer_score.numeric_match,
                    "semantic_similarity": r.answer_score.semantic_similarity,
                    "llm_judge_score": r.answer_score.llm_judge_score,
                    "reasoning": r.answer_score.reasoning,
                },
                "retrieved_chunks": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "document": c.get("document"),
                        "page": c.get("page"),
                        "distance": c.get("distance"),
                        "snippet": str(c.get("text", ""))[:200] + "...",
                    }
                    for c in r.retrieved_chunks
                ],
            })

        return {
            "summary": self.summary,
            "samples": detailed_samples,
        }

    def save_json(self, output_dir: str = "benchmark_runs") -> Path:
        """Save benchmark run details to a JSON file."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"{self.dataset_name.lower()}_eval_{self.timestamp}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        return file_path

    def save_markdown(self, output_dir: str = "benchmark_runs") -> Path:
        """Save human-readable Markdown summary report."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"{self.dataset_name.lower()}_summary_{self.timestamp}.md"

        ret = self.summary["retrieval"]
        ans = self.summary["answer_quality"]

        md_content = f"""# {self.dataset_name.upper()} Benchmark Report

- **Date / Timestamp**: `{self.timestamp}`
- **Total Evaluated Questions**: `{self.summary['total_samples']}`
- **Total Evaluation Time**: `{self.summary['total_time_seconds']}s`
- **Average Query Latency**: `{self.summary['average_latency_seconds']}s`

---

## 1. Retrieval Performance

| Metric | Score | Percentage |
| :--- | :--- | :--- |
| **Document Hit Rate** | `{ret['document_hit_rate']}` | `{ret['document_hit_rate']*100:.1f}%` |
| **Page Hit Rate (Target Page)** | `{ret['page_hit_rate']}` | `{ret['page_hit_rate']*100:.1f}%` |
| **Hit@1** | `{ret['hit_at_1']}` | `{ret['hit_at_1']*100:.1f}%` |
| **Hit@3** | `{ret['hit_at_3']}` | `{ret['hit_at_3']*100:.1f}%` |
| **Hit@5** | `{ret['hit_at_5']}` | `{ret['hit_at_5']*100:.1f}%` |
| **Mean Reciprocal Rank (MRR)** | `{ret['mrr']:.4f}` | - |
| **Average Text Overlap** | `{ret['average_text_overlap']:.4f}` | - |

---

## 2. Answer Quality

| Metric | Score | Percentage |
| :--- | :--- | :--- |
| **Overall Answer Accuracy** | `{ans['accuracy']}` | **`{ans['accuracy']*100:.1f}%`** |
| **Exact Match Rate** | `{ans['exact_match_rate']}` | `{ans['exact_match_rate']*100:.1f}%` |
| **Numeric Match Rate** | `{ans['numeric_match_rate']}` | `{ans['numeric_match_rate']*100:.1f}%` |
| **Avg Semantic Similarity** | `{ans['average_semantic_similarity']:.4f}` | `{ans['average_semantic_similarity']*100:.1f}%` |
| **LLM Judge Accuracy** | `{ans['llm_judge_accuracy'] if ans['llm_judge_accuracy'] is not None else 'N/A'}` | `{f"{ans['llm_judge_accuracy']*100:.1f}%" if ans['llm_judge_accuracy'] is not None else 'N/A'}` |

---

## 3. Sample Breakdown (First 15 Examples)

| ID | Company | Question | Gold Answer | Generated Answer | Ret. Page Hit | Correct |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for r in self.results[:15]:
            q_snippet = (r.sample.question[:45] + "...") if len(r.sample.question) > 45 else r.sample.question
            gold_snippet = (r.sample.answer[:30] + "...") if len(r.sample.answer) > 30 else r.sample.answer
            gen_snippet = (r.generated_answer[:30] + "...") if len(r.generated_answer) > 30 else r.generated_answer
            page_hit_str = "✅" if r.retrieval_score.page_hit else "❌"
            correct_str = "✅" if r.answer_score.is_correct else "❌"

            # Escape pipes
            q_snippet = q_snippet.replace("|", "/")
            gold_snippet = gold_snippet.replace("|", "/")
            gen_snippet = gen_snippet.replace("|", "/")

            md_content += f"| `{r.sample.id}` | {r.sample.company or 'N/A'} | {q_snippet} | {gold_snippet} | {gen_snippet} | {page_hit_str} | {correct_str} |\n"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return file_path

    def print_summary(self):
        """Print an ASCII summary table to console."""
        ret = self.summary["retrieval"]
        ans = self.summary["answer_quality"]

        print("\n" + "=" * 70)
        print(f"       BENCHMARK EVALUATION SUMMARY: {self.dataset_name.upper()}")
        print("=" * 70)
        print(f"Total Samples: {self.summary['total_samples']} | Total Time: {self.summary['total_time_seconds']}s | Avg Latency: {self.summary['average_latency_seconds']}s")
        print("-" * 70)
        print("RETRIEVAL METRICS:")
        print(f"  - Hit@1:                {ret['hit_at_1']*100:6.2f}%")
        print(f"  - Hit@3:                {ret['hit_at_3']*100:6.2f}%")
        print(f"  - Hit@5:                {ret['hit_at_5']*100:6.2f}%")
        print(f"  - Page Hit Rate:        {ret['page_hit_rate']*100:6.2f}%")
        print(f"  - Document Hit Rate:    {ret['document_hit_rate']*100:6.2f}%")
        print(f"  - Mean Reciprocal Rank: {ret['mrr']:6.4f}")
        print(f"  - Text Overlap Score:   {ret['average_text_overlap']:6.4f}")
        print("-" * 70)
        print("ANSWER QUALITY METRICS:")
        print(f"  - Overall Accuracy:     {ans['accuracy']*100:6.2f}%")
        print(f"  - Exact Match Rate:     {ans['exact_match_rate']*100:6.2f}%")
        print(f"  - Numeric Match Rate:   {ans['numeric_match_rate']*100:6.2f}%")
        print(f"  - Semantic Similarity:  {ans['average_semantic_similarity']*100:6.2f}%")
        if ans["llm_judge_accuracy"] is not None:
            print(f"  - LLM Judge Accuracy:   {ans['llm_judge_accuracy']*100:6.2f}%")
        print("=" * 70 + "\n")
