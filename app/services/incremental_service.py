from app.database.chroma_db import ChromaDBClient
from app.services.embedding_service import EmbeddingService


class IncrementalService:

    def __init__(self):

        self.chroma = ChromaDBClient()

        self.embedder = EmbeddingService()

    def process_changes(

        self,

        changes,

        chunks,

        filename

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

            chunk_id = f"{filename}_{chunk['chunk_number']}"

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

            chunk_id = f"{filename}_{chunk['chunk_number']}"

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

            chunk_id = f"{filename}_{index+1}"

            self.chroma.delete_chunk(

                chunk_id

            )

            stats["deleted"] += 1

        return stats