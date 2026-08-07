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

    def __init__(self):

        self.loader = PDFLoader()

        self.cleaner = TextCleaner()

        self.splitter = FinancialTextSplitter()

        self.embedder = EmbeddingService()

        self.chroma = ChromaDBClient()

        self.sqlite = ManifestDB()

        self.manifest = ManifestService()

        self.incremental = IncrementalService()

        self.analytics = AnalyticsService()

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

            filename

        )

        # -------------------------
        # Update Manifest
        # -------------------------

        document_id = filename

        for i, chunk in enumerate(chunks):

            chunk_id = f"{filename}_{chunk['chunk_number']}"

            if i in changes["new"]:

                self.manifest.add_chunk(

                    document_name=filename,

                    document_id=document_id,

                    chunk_id=chunk_id,

                    chunk_index=i,

                    page=chunk["page"],

                    chunk_hash=hashes[i],

                    embedding_id=chunk_id

                )

            elif i in changes["modified"]:

                self.manifest.update_chunk(

                    chunk_id,

                    hashes[i],

                    chunk_id

                )

        # -------------------------
        # Remove Deleted
        # -------------------------

        for index in changes["deleted"]:

            chunk_id = f"{filename}_{index+1}"

            self.manifest.delete_chunk(

                chunk_id

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