from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional
from pathlib import Path
import math

from app.eval.base import BenchmarkSample


@dataclass
class RoundPartition:
    """Represents a sequential incremental update batch/round."""
    round_index: int
    batch_docs: List[str]
    cumulative_docs: List[str]
    new_queries: List[BenchmarkSample]
    cumulative_queries: List[BenchmarkSample]


@dataclass
class CorpusSplit:
    """Contains the base/update corpus split and categorized benchmark queries."""
    all_docs: List[str]
    base_docs: List[str]
    update_docs: List[str]
    base_queries: List[BenchmarkSample]
    new_queries: List[BenchmarkSample]
    all_queries: List[BenchmarkSample]
    round_partitions: List[RoundPartition] = field(default_factory=list)

    @property
    def base_doc_count(self) -> int:
        return len(self.base_docs)

    @property
    def update_doc_count(self) -> int:
        return len(self.update_docs)

    @property
    def base_query_count(self) -> int:
        return len(self.base_queries)

    @property
    def new_query_count(self) -> int:
        return len(self.new_queries)


class CorpusSplitter:
    """
    Splits a benchmark corpus and associated QA dataset into:
    - Base Corpus (indexed first, established baseline)
    - Update Corpus (indexed incrementally across 1 or more sequential rounds)
    - Base Queries (answerable strictly from base corpus)
    - New Queries (introduced with and require incremental document updates)
    Following ECT-QA (arXiv:2510.13590) & EraRAG (arXiv:2506.20963) methodology.
    """

    @staticmethod
    def _normalize_doc_name(doc_name: str) -> str:
        if not doc_name:
            return ""
        name = str(doc_name).strip()
        if name.lower().endswith(".pdf"):
            name = name[:-4]
        return name

    @classmethod
    def get_sample_required_docs(cls, sample: BenchmarkSample) -> Set[str]:
        """Extract all unique normalized document names required to answer a sample."""
        docs: Set[str] = set()
        if sample.doc_name:
            docs.add(cls._normalize_doc_name(sample.doc_name))
        for ev in sample.evidence:
            if ev.doc_name:
                docs.add(cls._normalize_doc_name(ev.doc_name))
        return docs

    @classmethod
    def split(
        cls,
        samples: List[BenchmarkSample],
        base_fraction: float = 0.8,
        rounds: int = 1,
    ) -> CorpusSplit:
        """
        Split samples and documents into base and update slices.

        Args:
            samples: List of benchmark question samples.
            base_fraction: Fraction of unique documents assigned to base corpus (0.0 < fraction < 1.0).
            rounds: Number of sequential incremental update batches (>= 1).
        """
        if not samples:
            return CorpusSplit(
                all_docs=[],
                base_docs=[],
                update_docs=[],
                base_queries=[],
                new_queries=[],
                all_queries=[],
                round_partitions=[],
            )

        # 1. Discover all unique documents in stable order
        doc_set: Set[str] = set()
        for sample in samples:
            doc_set.update(cls.get_sample_required_docs(sample))
        all_docs = sorted(list(doc_set))

        num_docs = len(all_docs)

        # 2. Partition documents into Base and Update sets
        if num_docs <= 1:
            base_docs = list(all_docs)
            update_docs = []
        else:
            base_count = int(math.floor(num_docs * base_fraction))
            # Ensure at least 1 base doc and at least 1 update doc when num_docs >= 2
            base_count = max(1, min(base_count, num_docs - 1))
            base_docs = all_docs[:base_count]
            update_docs = all_docs[base_count:]

        base_docs_set = set(base_docs)
        update_docs_set = set(update_docs)

        # 3. Categorize queries into Base vs New
        base_queries: List[BenchmarkSample] = []
        new_queries: List[BenchmarkSample] = []

        for sample in samples:
            req_docs = cls.get_sample_required_docs(sample)
            if not req_docs:
                # If no doc specified, assign to base
                base_queries.append(sample)
            elif req_docs.issubset(base_docs_set):
                base_queries.append(sample)
            else:
                new_queries.append(sample)

        # 4. Multi-round partitioning of update documents
        num_rounds = max(1, rounds)
        round_partitions: List[RoundPartition] = []

        if not update_docs:
            # Single base-only round partition
            round_partitions.append(
                RoundPartition(
                    round_index=0,
                    batch_docs=base_docs,
                    cumulative_docs=base_docs,
                    new_queries=base_queries,
                    cumulative_queries=base_queries,
                )
            )
        else:
            # Distribute update_docs across num_rounds non-empty batches
            effective_rounds = min(num_rounds, len(update_docs))
            batch_size = math.ceil(len(update_docs) / effective_rounds)

            update_batches: List[List[str]] = []
            for i in range(0, len(update_docs), batch_size):
                update_batches.append(update_docs[i : i + batch_size])

            # Round 0: Base partition
            round_partitions.append(
                RoundPartition(
                    round_index=0,
                    batch_docs=base_docs,
                    cumulative_docs=base_docs,
                    new_queries=base_queries,
                    cumulative_queries=base_queries,
                )
            )

            # Sequential incremental rounds
            cumulative_docs = list(base_docs)
            cumulative_queries = list(base_queries)
            prev_docs_set = set(base_docs)

            for r_idx, batch in enumerate(update_batches, start=1):
                cumulative_docs = cumulative_docs + batch
                current_docs_set = set(cumulative_docs)

                # Find new queries activated in this round
                round_new_queries: List[BenchmarkSample] = []
                for sample in new_queries:
                    req_docs = cls.get_sample_required_docs(sample)
                    # Activated if all required docs are now in current_docs_set but were not in prev_docs_set
                    if req_docs.issubset(current_docs_set) and not req_docs.issubset(prev_docs_set):
                        round_new_queries.append(sample)

                cumulative_queries = cumulative_queries + round_new_queries
                prev_docs_set = current_docs_set

                round_partitions.append(
                    RoundPartition(
                        round_index=r_idx,
                        batch_docs=batch,
                        cumulative_docs=cumulative_docs,
                        new_queries=round_new_queries,
                        cumulative_queries=cumulative_queries,
                    )
                )

        return CorpusSplit(
            all_docs=all_docs,
            base_docs=base_docs,
            update_docs=update_docs,
            base_queries=base_queries,
            new_queries=new_queries,
            all_queries=samples,
            round_partitions=round_partitions,
        )
