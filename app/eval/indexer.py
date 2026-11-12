import time
from pathlib import Path
from typing import List, Dict, Any, Union
from tqdm import tqdm

from app.services.document_service import DocumentProcessingService


class BenchmarkIndexer:
    """Indexes benchmark documents using the project's DocumentProcessingService."""

    def __init__(
        self,
        manifest_db_path: str = "manifest_db/manifest.db",
        chroma_path: str = "chroma_storage",
        chroma_collection_name: str = "financebench_docs",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.manifest_db_path = manifest_db_path
        self.chroma_path = chroma_path
        self.chroma_collection_name = chroma_collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.service = DocumentProcessingService(
            manifest_db_path=self.manifest_db_path,
            chroma_path=self.chroma_path,
            chroma_collection_name=self.chroma_collection_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def index_documents(
        self,
        pdf_paths: List[Union[str, Path]],
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest and index a list of PDF documents.
        Uses incremental indexing under the hood so unchanged documents are skipped.
        """
        start_time = time.perf_counter()
        total_pages = 0
        total_chunks = 0
        total_embedding_calls = 0
        total_saved_calls = 0
        doc_summaries = []

        iterator = tqdm(pdf_paths, desc="Indexing Benchmark PDFs") if show_progress else pdf_paths

        for pdf_path in iterator:
            path_obj = Path(pdf_path)
            if not path_obj.exists():
                raise FileNotFoundError(f"PDF document not found: {path_obj}")

            result = self.service.process_document(str(path_obj))
            doc_summaries.append(result)

            total_pages += result.get("pages", 0)
            total_chunks += result.get("chunks", 0)
            total_embedding_calls += result.get("embedding_calls", 0)
            total_saved_calls += result.get("embedding_calls_saved", 0)

        total_time = round(time.perf_counter() - start_time, 3)

        return {
            "total_documents": len(pdf_paths),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "total_embedding_calls": total_embedding_calls,
            "total_saved_calls": total_saved_calls,
            "total_indexing_time_seconds": total_time,
            "document_results": doc_summaries,
            "chroma_collection": self.chroma_collection_name,
            "chroma_path": self.chroma_path,
            "manifest_db": self.manifest_db_path,
        }
