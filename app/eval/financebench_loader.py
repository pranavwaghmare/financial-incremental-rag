import os
import json
import logging
import requests
from pathlib import Path
from typing import List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

from app.eval.base import BaseBenchmarkDataset, BenchmarkSample, EvidenceItem

logger = logging.getLogger(__name__)

FINANCEBENCH_JSONL_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl"
FINANCEBENCH_PDF_BASE_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"


class FinanceBenchDataset(BaseBenchmarkDataset):
    """Loader and manager for the Patronus AI FinanceBench open-source dataset."""

    def __init__(
        self,
        data_dir: str = "data/financebench",
        jsonl_url: str = FINANCEBENCH_JSONL_URL,
        pdf_base_url: str = FINANCEBENCH_PDF_BASE_URL,
    ):
        self.data_dir = Path(data_dir)
        self.pdf_dir = self.data_dir / "pdfs"
        self.jsonl_path = self.data_dir / "financebench_open_source.jsonl"
        self.jsonl_url = jsonl_url
        self.pdf_base_url = pdf_base_url

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset_if_needed(self, force: bool = False) -> Path:
        """Download financebench_open_source.jsonl if not present locally."""
        if not self.jsonl_path.exists() or force:
            print(f"Downloading FinanceBench dataset from {self.jsonl_url}...")
            response = requests.get(self.jsonl_url, timeout=30)
            response.raise_for_status()
            with open(self.jsonl_path, "wb") as f:
                f.write(response.content)
            print(f"FinanceBench dataset saved to {self.jsonl_path}")
        return self.jsonl_path

    def load(self, n: Optional[int] = None) -> List[BenchmarkSample]:
        """Load benchmark samples from local JSONL file (downloading if needed)."""
        self.download_dataset_if_needed()

        samples: List[BenchmarkSample] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                evidence_list = []
                for ev in data.get("evidence", []):
                    evidence_list.append(
                        EvidenceItem(
                            evidence_text=ev.get("evidence_text", ""),
                            evidence_page_num=ev.get("evidence_page_num", 0),
                            doc_name=ev.get("doc_name", data.get("doc_name", "")),
                            evidence_text_full_page=ev.get("evidence_text_full_page"),
                        )
                    )

                sample = BenchmarkSample(
                    id=data.get("financebench_id", ""),
                    question=data.get("question", ""),
                    answer=data.get("answer", ""),
                    doc_name=data.get("doc_name", ""),
                    company=data.get("company"),
                    question_type=data.get("question_type"),
                    question_reasoning=data.get("question_reasoning"),
                    justification=data.get("justification"),
                    evidence=evidence_list,
                    metadata={
                        "domain_question_num": data.get("domain_question_num"),
                        "dataset_subset_label": data.get("dataset_subset_label"),
                    },
                )
                samples.append(sample)
                if n is not None and len(samples) >= n:
                    break

        return samples

    def get_required_documents(self, samples: List[BenchmarkSample]) -> List[str]:
        """Return list of distinct document names needed for the samples."""
        docs: Set[str] = set()
        for sample in samples:
            if sample.doc_name:
                docs.add(sample.doc_name)
            for ev in sample.evidence:
                if ev.doc_name:
                    docs.add(ev.doc_name)
        return sorted(list(docs))

    def get_pdf_path(self, doc_name: str) -> Path:
        """Get local filepath for a given document name."""
        filename = doc_name if doc_name.endswith(".pdf") else f"{doc_name}.pdf"
        return self.pdf_dir / filename

    def download_pdf(self, doc_name: str, force: bool = False) -> Path:
        """Download a single PDF from the FinanceBench GitHub repository."""
        target_path = self.get_pdf_path(doc_name)
        if target_path.exists() and not force:
            return target_path

        filename = target_path.name
        encoded_filename = quote(filename)
        url = f"{self.pdf_base_url}/{encoded_filename}"

        response = requests.get(url, stream=True, timeout=60)
        if response.status_code != 200:
            # Fallback to direct raw github URL format without quote
            url_fallback = f"https://github.com/patronus-ai/financebench/raw/main/pdfs/{encoded_filename}"
            response = requests.get(url_fallback, stream=True, timeout=60)

        response.raise_for_status()

        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return target_path

    def download_pdfs_for_samples(
        self,
        samples: List[BenchmarkSample],
        max_workers: int = 5,
        force: bool = False,
    ) -> List[Path]:
        """Download all PDFs required by the given benchmark samples."""
        doc_names = self.get_required_documents(samples)
        pdf_paths: List[Path] = []
        needed_docs = [doc for doc in doc_names if not self.get_pdf_path(doc).exists() or force]

        if needed_docs:
            print(f"Downloading {len(needed_docs)} required PDF documents for evaluation...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_doc = {
                    executor.submit(self.download_pdf, doc, force): doc
                    for doc in needed_docs
                }
                for future in as_completed(future_to_doc):
                    doc = future_to_doc[future]
                    try:
                        path = future.result()
                        pdf_paths.append(path)
                    except Exception as e:
                        print(f"Error downloading PDF for {doc}: {e}")
                        raise

        for doc in doc_names:
            pdf_paths.append(self.get_pdf_path(doc))

        # Unique paths
        unique_paths = sorted(list(set(pdf_paths)), key=lambda p: p.name)
        return unique_paths
