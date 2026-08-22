from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


@dataclass
class EvidenceItem:
    evidence_text: str
    evidence_page_num: int  # 0-indexed from benchmark (e.g. FinanceBench)
    doc_name: str
    evidence_text_full_page: Optional[str] = None

    @property
    def page_number_1_indexed(self) -> int:
        """Return 1-indexed page number matching PDF page numbering in loaders."""
        return self.evidence_page_num + 1


@dataclass
class BenchmarkSample:
    id: str
    question: str
    answer: str
    doc_name: str
    company: Optional[str] = None
    question_type: Optional[str] = None
    question_reasoning: Optional[str] = None
    justification: Optional[str] = None
    evidence: List[EvidenceItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalScore:
    doc_hit: bool = False
    page_hit: bool = False
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    mrr: float = 0.0
    text_overlap_score: float = 0.0
    gold_pages_1_indexed: List[int] = field(default_factory=list)
    retrieved_pages_1_indexed: List[int] = field(default_factory=list)
    retrieved_docs: List[str] = field(default_factory=list)


@dataclass
class AnswerScore:
    is_correct: bool = False
    overall_score: float = 0.0
    exact_match: bool = False
    numeric_match: bool = False
    semantic_similarity: float = 0.0
    llm_judge_score: Optional[float] = None
    reasoning: str = ""


@dataclass
class SampleEvaluationResult:
    sample: BenchmarkSample
    generated_answer: str
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_score: RetrievalScore
    answer_score: AnswerScore
    latency_seconds: float = 0.0


class BaseBenchmarkDataset(ABC):
    """Abstract base class for financial QA benchmark datasets (FinanceBench, FinQA, TAT-QA, etc.)."""

    @abstractmethod
    def load(self, n: Optional[int] = None) -> List[BenchmarkSample]:
        """Load and return benchmark samples."""
        pass

    @abstractmethod
    def get_required_documents(self, samples: List[BenchmarkSample]) -> List[str]:
        """Return unique document names needed for the given samples."""
        pass
