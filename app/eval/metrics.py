import re
import json
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from app.eval.base import BenchmarkSample, RetrievalScore, AnswerScore
from app.services.embedding_service import EmbeddingService
from app.services.groq_service import GroqService


class RetrievalEvaluator:
    """Evaluates retrieval quality against ground-truth evidence."""

    @staticmethod
    def _normalize_doc_name(doc_name: str) -> str:
        """Normalize document name by removing extension and lowercasing."""
        if not doc_name:
            return ""
        name = str(doc_name).strip()
        if name.lower().endswith(".pdf"):
            name = name[:-4]
        return name.lower()

    @classmethod
    def evaluate_retrieval(
        cls,
        retrieved_chunks: List[Dict[str, Any]],
        sample: BenchmarkSample,
        top_k: int = 5,
    ) -> RetrievalScore:
        gold_doc_names: Set[str] = {
            cls._normalize_doc_name(sample.doc_name)
        }
        gold_pages_1_indexed: Set[int] = set()

        for ev in sample.evidence:
            if ev.doc_name:
                gold_doc_names.add(cls._normalize_doc_name(ev.doc_name))
            gold_pages_1_indexed.add(ev.page_number_1_indexed)

        retrieved_docs: List[str] = []
        retrieved_pages: List[int] = []
        doc_hit = False
        page_hit = False
        hit_at_1 = False
        hit_at_3 = False
        hit_at_5 = False
        mrr = 0.0
        max_text_overlap = 0.0

        for rank, chunk in enumerate(retrieved_chunks[:top_k], start=1):
            chunk_doc = cls._normalize_doc_name(chunk.get("document", ""))
            chunk_page = chunk.get("page", 0)
            try:
                chunk_page_int = int(chunk_page)
            except (ValueError, TypeError):
                chunk_page_int = -1

            retrieved_docs.append(chunk.get("document", ""))
            retrieved_pages.append(chunk_page_int)

            is_doc_match = chunk_doc in gold_doc_names
            if is_doc_match:
                doc_hit = True

            # Page match requires matching the target document and page number
            is_page_match = is_doc_match and (chunk_page_int in gold_pages_1_indexed)
            if is_page_match:
                page_hit = True
                if mrr == 0.0:
                    mrr = 1.0 / rank
                if rank == 1:
                    hit_at_1 = True
                if rank <= 3:
                    hit_at_3 = True
                if rank <= 5:
                    hit_at_5 = True

            # Text overlap score (token containment with evidence text)
            chunk_text = str(chunk.get("text", "")).lower()
            for ev in sample.evidence:
                ev_text = ev.evidence_text.lower().strip()
                if ev_text:
                    # Check token overlap
                    ev_tokens = set(re.findall(r"\w+", ev_text))
                    chunk_tokens = set(re.findall(r"\w+", chunk_text))
                    if ev_tokens:
                        overlap = len(ev_tokens & chunk_tokens) / len(ev_tokens)
                        if overlap > max_text_overlap:
                            max_text_overlap = overlap

        return RetrievalScore(
            doc_hit=doc_hit,
            page_hit=page_hit,
            hit_at_1=hit_at_1,
            hit_at_3=hit_at_3,
            hit_at_5=hit_at_5,
            mrr=mrr,
            text_overlap_score=round(max_text_overlap, 4),
            gold_pages_1_indexed=sorted(list(gold_pages_1_indexed)),
            retrieved_pages_1_indexed=retrieved_pages,
            retrieved_docs=retrieved_docs,
        )


class AnswerEvaluator:
    """Evaluates answer quality using exact match, numeric matching, semantic similarity, and LLM-as-a-judge."""

    def __init__(
        self,
        embedder: Optional[EmbeddingService] = None,
        groq_service: Optional[GroqService] = None,
    ):
        self.embedder = embedder or EmbeddingService()
        self.groq_service = groq_service

    @staticmethod
    def normalize_text(text: str) -> str:
        """Standard text normalization."""
        if not text:
            return ""
        text = text.lower().strip()
        # Remove markdown symbols and extra punctuation
        text = re.sub(r"[\*\_\[\]\(\)\,\$\%]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """Extract numeric values including decimals, negatives in parentheses, and percentages."""
        if not text:
            return []
        
        # Remove commas inside numbers (e.g. 1,577.00 -> 1577.00)
        processed = re.sub(r"(?<=\d),(?=\d)", "", text)
        # Handle (123.45) as -123.45
        processed = re.sub(r"\(([\d\.]+)\)", r"-\1", processed)
        
        # Find numbers
        matches = re.findall(r"[-+]?\d*\.?\d+", processed)
        numbers = []
        for m in matches:
            try:
                numbers.append(float(m))
            except ValueError:
                pass
        return numbers

    def check_numeric_match(self, generated_answer: str, gold_answer: str) -> bool:
        """Check if numbers in the gold answer appear in the generated answer within tolerance."""
        gold_nums = self.extract_numbers(gold_answer)
        if not gold_nums:
            return False

        gen_nums = self.extract_numbers(generated_answer)
        if not gen_nums:
            return False

        # Check if every key gold number is represented in generated numbers
        matches = 0
        for g_num in gold_nums:
            matched = False
            for gen_num in gen_nums:
                # Direct match or absolute value match (e.g. 'decreased by 1.7%' vs '-1.7%')
                if abs(g_num - gen_num) < 1e-3 or abs(abs(g_num) - abs(gen_num)) < 1e-3:
                    matched = True
                    break
                # Check percentage/multiplier scales (x100, x1000, x1M)
                if abs(abs(g_num) * 100 - abs(gen_num)) < 1e-2 or abs(abs(gen_num) * 100 - abs(g_num)) < 1e-2:
                    matched = True
                    break
                if abs(abs(g_num) * 1000 - abs(gen_num)) < 1e-1 or abs(abs(gen_num) * 1000 - abs(g_num)) < 1e-1:
                    matched = True
                    break
                if g_num != 0 and abs((abs(gen_num) - abs(g_num)) / abs(g_num)) <= 0.03:  # 3% tolerance
                    matched = True
                    break
            if matched:
                matches += 1

        return matches >= max(1, len(gold_nums) // 2)

    def check_exact_match(self, generated_answer: str, gold_answer: str) -> bool:
        """Check exact normalized match or full substring containment."""
        norm_gen = self.normalize_text(generated_answer)
        norm_gold = self.normalize_text(gold_answer)

        if not norm_gen or not norm_gold:
            return False

        if norm_gen == norm_gold or norm_gold in norm_gen:
            return True
        return False

    def compute_semantic_similarity(self, generated_answer: str, gold_answer: str) -> float:
        """Compute cosine similarity between generated answer and gold answer embeddings."""
        if not generated_answer or not gold_answer:
            return 0.0

        vec_gen = np.array(self.embedder.embed_text(generated_answer))
        vec_gold = np.array(self.embedder.embed_text(gold_answer))

        norm_gen = np.linalg.norm(vec_gen)
        norm_gold = np.linalg.norm(vec_gold)

        if norm_gen == 0 or norm_gold == 0:
            return 0.0

        sim = float(np.dot(vec_gen, vec_gold) / (norm_gen * norm_gold))
        return round(max(0.0, min(1.0, sim)), 4)

    def run_llm_judge(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
        justification: Optional[str] = None,
    ) -> Tuple[bool, float, str]:
        """Run Groq LLM-as-a-judge evaluation."""
        if not self.groq_service:
            return False, 0.0, "LLM judge unavailable (GroqService not configured)"

        prompt = f"""You are an expert financial auditor evaluating the answer of a Financial RAG assistant.

Question:
{question}

Gold Reference Answer:
{gold_answer}

Gold Justification / Details:
{justification or "N/A"}

Candidate Answer:
{generated_answer}

Instructions:
Evaluate if the Candidate Answer is factually accurate and answers the question correctly according to the Gold Reference Answer.
- Minor variations in wording are acceptable as long as numbers, directionality, and core facts match.
- If the Candidate Answer contains wrong numbers, hallucinated facts, or contradicts the gold answer, mark is_correct as false.

Respond ONLY with a valid JSON object in this exact format:
{{"is_correct": true, "score": 1.0, "reasoning": "Explanation here"}}
"""
        try:
            raw_response = self.groq_service.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are an objective evaluation system. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_completion_tokens=256,
            )
            content = raw_response.choices[0].message.content.strip()
            # Clean possible markdown wrapping ```json ... ```
            clean_json = re.sub(r"^```json\s*", "", content)
            clean_json = re.sub(r"\s*```$", "", clean_json)
            parsed = json.loads(clean_json)

            is_correct = bool(parsed.get("is_correct", False))
            score = float(parsed.get("score", 1.0 if is_correct else 0.0))
            reasoning = str(parsed.get("reasoning", ""))
            return is_correct, score, reasoning
        except Exception as e:
            # Fallback
            return False, 0.0, f"LLM judge evaluation error: {str(e)}"

    def evaluate(
        self,
        generated_answer: str,
        sample: BenchmarkSample,
        mode: str = "hybrid",  # 'hybrid', 'semantic', 'llm', 'numeric'
    ) -> AnswerScore:
        """Comprehensive evaluation of the candidate answer."""
        gold_answer = sample.answer
        exact = self.check_exact_match(generated_answer, gold_answer)
        numeric = self.check_numeric_match(generated_answer, gold_answer)
        semantic_sim = self.compute_semantic_similarity(generated_answer, gold_answer)

        llm_correct = None
        llm_score = None
        llm_reasoning = ""

        if mode in ("llm", "hybrid") and self.groq_service is not None:
            try:
                llm_correct, llm_score, llm_reasoning = self.run_llm_judge(
                    question=sample.question,
                    gold_answer=gold_answer,
                    generated_answer=generated_answer,
                    justification=sample.justification,
                )
            except Exception as e:
                llm_reasoning = f"LLM judge skipped: {e}"

        # Determine overall correctness
        if mode == "semantic":
            is_correct = semantic_sim >= 0.80 or exact or numeric
            overall_score = semantic_sim
            reasoning = f"Semantic similarity: {semantic_sim:.3f}"
        elif mode == "numeric":
            is_correct = numeric or exact
            overall_score = 1.0 if is_correct else 0.0
            reasoning = f"Numeric match: {numeric}, Exact match: {exact}"
        elif mode == "llm" and llm_correct is not None:
            is_correct = llm_correct
            overall_score = llm_score if llm_score is not None else (1.0 if is_correct else 0.0)
            reasoning = llm_reasoning
        else:
            # Hybrid mode: combines LLM judge if available, with strong semantic / numeric fallback
            if llm_correct is not None:
                is_correct = llm_correct or exact or (numeric and semantic_sim >= 0.70)
                overall_score = llm_score if llm_correct else max(semantic_sim, 0.0)
                reasoning = llm_reasoning or f"Sim: {semantic_sim:.2f}, Numeric: {numeric}"
            else:
                is_correct = exact or numeric or semantic_sim >= 0.80
                overall_score = max(float(exact), float(numeric), semantic_sim)
                reasoning = f"Sim: {semantic_sim:.2f}, Numeric: {numeric}, Exact: {exact}"

        return AnswerScore(
            is_correct=is_correct,
            overall_score=round(overall_score, 4),
            exact_match=exact,
            numeric_match=numeric,
            semantic_similarity=semantic_sim,
            llm_judge_score=llm_score,
            reasoning=reasoning,
        )
