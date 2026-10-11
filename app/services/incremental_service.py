from app.database.chroma_db import ChromaDBClient
from app.services.embedding_service import EmbeddingService
from app.services.hashing_service import HashingService


class IncrementalService:

    def __init__(self, chroma_client=None):

        self.chroma = chroma_client or ChromaDBClient()

        self.embedder = EmbeddingService()

    def process_changes(

        self,

        changes,

        chunks,

        filename,

        old_hashes=None,

        new_hashes=None

    ):

        stats = {

            "new": 0,

            "modified": 0,

            "deleted": 0,

            "unchanged": len(changes["unchanged"])

        }

        # NEW

        for index in changes["new"]:

            chunk = chunks[index]

            embedding = self.embedder.embed_text(

                chunk["text"]

            )

            chunk_hash = (new_hashes or {}).get(index) or HashingService.generate_hash(chunk["text"])
            chunk_id = HashingService.chunk_id(filename, chunk_hash)

            self.chroma.add_chunk(

                chunk_id,

                chunk["text"],

                embedding,

                {

                    "document_name": filename,

                    "page": chunk["page"],

                    "chunk_number": chunk["chunk_number"]

                }

            )

            stats["new"] += 1

        # MODIFIED

        for index in changes["modified"]:

            chunk = chunks[index]

            embedding = self.embedder.embed_text(

                chunk["text"]

            )

            chunk_hash = (new_hashes or {}).get(index) or HashingService.generate_hash(chunk["text"])
            chunk_id = HashingService.chunk_id(filename, chunk_hash)

            self.chroma.update_chunk(

                chunk_id,

                chunk["text"],

                embedding,

                {

                    "document_name": filename,

                    "page": chunk["page"],

                    "chunk_number": chunk["chunk_number"]

                }

            )

            stats["modified"] += 1

        # DELETED

        for index in changes["deleted"]:

            chunk_hash = (old_hashes or {}).get(index)
            if chunk_hash is None:
                continue
            chunk_id = HashingService.chunk_id(filename, chunk_hash)

            self.chroma.delete_chunk(

                chunk_id

            )

            stats["deleted"] += 1

        return stats