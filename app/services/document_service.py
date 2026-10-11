from pathlib import Path
import time
from app.database.sqlite_db import ManifestDB
from app.database.chroma_db import ChromaDBClient

from app.services.embedding_service import EmbeddingService
from app.services.hashing_service import HashingService
from app.services.manifest_service import ManifestService
from app.services.diff_service import DiffService
from app.services.incremental_service import IncrementalService
from app.services.analytics_service import AnalyticsService

from app.utils.cleaner import TextCleaner
from app.utils.pdf_loader import PDFLoader
from app.utils.splitter import FinancialTextSplitter


class DocumentProcessingService:

    def __init__(
        self,
        manifest_db_path: str = "manifest_db/manifest.db",
        chroma_path: str = "chroma_storage",
        chroma_collection_name: str = "financial_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):

        self.loader = PDFLoader()

        self.cleaner = TextCleaner()

        self.splitter = FinancialTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.embedder = EmbeddingService()

        self.chroma = ChromaDBClient(path=chroma_path, collection_name=chroma_collection_name)

        self.sqlite = ManifestDB(db_path=manifest_db_path)

        self.manifest = ManifestService(db_path=manifest_db_path)

        self.incremental = IncrementalService(chroma_client=self.chroma)

        self.analytics = AnalyticsService()

    def process_document_full_rebuild(self, file_path):
        start_time = time.perf_counter()
        filename = Path(file_path).name

        pages = self.loader.extract_text(file_path)

        cleaned_pages = [
            {"page": page["page"], "text": self.cleaner.clean(page["text"])}
            for page in pages
        ]

        chunks = self.splitter.split_pages(cleaned_pages)

        embedding_calls = 0
        for chunk in chunks:
            embedding = self.embedder.embed_text(chunk["text"])
            embedding_calls += 1

            chunk_hash = HashingService.generate_hash(chunk["text"])
            chunk_id = HashingService.chunk_id(filename, chunk_hash)
            self.chroma.collection.upsert(
                ids=[chunk_id],
                documents=[chunk["text"]],
                embeddings=[embedding],
                metadatas=[{
                    "document_name": filename,
                    "page": chunk["page"],
                    "chunk_number": chunk["chunk_number"],
                }],
            )

        end_time = time.perf_counter()
        processing_time = round(end_time - start_time, 3)

        return {
            "filename": filename,
            "chunks": len(chunks),
            "embedding_calls": embedding_calls,
            "processing_time_seconds": processing_time,
            "mode": "full_rebuild",
        }

    def benchmark_against_full_rebuild(self, version_1_path, version_2_path, temp_root=None, chunk_size=1000, chunk_overlap=200):
        from tempfile import TemporaryDirectory
        import shutil

        if temp_root is not None:
            temp_dir = Path(temp_root)
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_root_obj = temp_dir
        else:
            temp_root_obj = Path(TemporaryDirectory(prefix="financial_rag_benchmark_").name)
            temp_root_obj.mkdir(parents=True, exist_ok=True)

        v1_copy = temp_root_obj / "v1.pdf"
        v2_copy = temp_root_obj / "v2.pdf"
        shutil.copy2(version_1_path, v1_copy)
        shutil.copy2(version_2_path, v2_copy)

        inc_path = temp_root_obj / "incremental"
        full_path = temp_root_obj / "full_rebuild"
        inc_path.mkdir(exist_ok=True)
        full_path.mkdir(exist_ok=True)

        incremental_service = DocumentProcessingService(
            manifest_db_path=str(inc_path / "manifest.db"),
            chroma_path=str(inc_path / "chroma"),
            chroma_collection_name="incremental_benchmark",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        full_service = DocumentProcessingService(
            manifest_db_path=str(full_path / "manifest.db"),
            chroma_path=str(full_path / "chroma"),
            chroma_collection_name="full_rebuild_benchmark",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        incremental_service.process_document(str(v1_copy))
        incremental_result = incremental_service.process_document(str(v2_copy))
        full_result = full_service.process_document_full_rebuild(str(v2_copy))

        result = {
            "incremental": incremental_result,
            "full_rebuild": full_result,
            "comparison": {
                "incremental_embedding_calls": incremental_result.get("embedding_calls", 0),
                "full_rebuild_embedding_calls": full_result.get("embedding_calls", 0),
                "incremental_less_than_full": incremental_result.get("embedding_calls", 0) < full_result.get("embedding_calls", 0),
            },
            "storage": {
                "incremental_manifest": str(inc_path / "manifest.db"),
                "incremental_chroma": str(inc_path / "chroma"),
                "full_manifest": str(full_path / "manifest.db"),
                "full_chroma": str(full_path / "chroma"),
            },
        }

        if temp_root is None:
            shutil.rmtree(temp_root_obj, ignore_errors=True)

        return result

    def process_document(self, file_path):
        start_time = time.perf_counter()
        filename = Path(file_path).name

        # -------------------------
        # Extract
        # -------------------------

        pages = self.loader.extract_text(file_path)

        # -------------------------
        # Clean
        # -------------------------

        cleaned_pages = [

            {

                "page": page["page"],

                "text": self.cleaner.clean(page["text"])

            }

            for page in pages

        ]

        # -------------------------
        # Split
        # -------------------------

        chunks = self.splitter.split_pages(

            cleaned_pages

        )

        # -------------------------
        # Hash
        # -------------------------

        texts = [

            chunk["text"]

            for chunk in chunks

        ]

        hashes = HashingService.generate_hashes(

            texts

        )

        new_hashes = {

            i: h

            for i, h in enumerate(hashes)

        }

        # The manifest stores chunk hashes keyed by their chunk index, but the
        # diff must be content-addressable: a chunk is unchanged if its hash value
        # exists anywhere in the previous version, regardless of index placement.
        # -------------------------
        # Read Manifest
        # -------------------------

        old_hashes = self.manifest.get_hash_map(

            filename

        )

        # -------------------------
        # Compare
        # -------------------------

        changes = DiffService.compare(

            old_hashes,

            new_hashes

        )

        # -------------------------
        # Update Chroma
        # -------------------------

        stats = self.incremental.process_changes(

            changes,

            chunks,

            filename,

            old_hashes=old_hashes,

            new_hashes=new_hashes

        )

        # -------------------------
        # Update Manifest
        # -------------------------

        document_id = filename

        # Reset the manifest for this document so index positions reflect the
        # current document layout, while diffing itself remains content-addressable.
        self.manifest.delete_document(filename)

        for i, chunk in enumerate(chunks):
            chunk_hash = hashes[i]
            chunk_id = HashingService.chunk_id(filename, chunk_hash)

            self.manifest.add_chunk(
                document_name=filename,
                document_id=document_id,
                chunk_id=chunk_id,
                chunk_index=i,
                page=chunk["page"],
                chunk_hash=chunk_hash,
                embedding_id=chunk_id,
            )

        # -------------------------
        # Store document metadata
        # -------------------------

        self.sqlite.insert_document(

            filename,

            len(pages),

            len(chunks)

        )

        reused_percentage = round(

            (stats["unchanged"] / len(chunks)) * 100,

            2

        ) if len(chunks) else 0

        end_time = time.perf_counter()
        processing_time = round(end_time - start_time, 3)

        if (
            stats["new"] == 0 and
            stats["modified"] == 0 and
            stats["deleted"] == 0
        ):

            message = "Document already indexed. No changes detected."

        else:

            message = "Incremental indexing completed."

        self.analytics.save_upload_metrics(

            filename=filename,

            company=filename.split("-")[0],

            sector="Unknown",

            pages=len(pages),

            chunks=len(chunks),

            new_chunks=stats["new"],

            modified_chunks=stats["modified"],

            deleted_chunks=stats["deleted"],

            unchanged_chunks=stats["unchanged"],

            embedding_calls=stats["new"] + stats["modified"],

            embedding_calls_saved=stats["unchanged"],

            reused_percentage=reused_percentage,

            processing_time=processing_time

        )
        # -------------------------
        # Response
        # -------------------------

        return {

            "filename": filename,

            "pages": len(pages),

            "chunks": len(chunks),

            "new_chunks": stats["new"],

            "modified_chunks": stats["modified"],

            "deleted_chunks": stats["deleted"],

            "unchanged_chunks": stats["unchanged"],

            "embedding_calls": stats["new"] + stats["modified"],

            "embedding_calls_saved": stats["unchanged"],

            "reused_chunks_percentage": reused_percentage,

            "processing_time_seconds": processing_time,

            "status": "Incremental Indexing Completed",

            "message": message

        }