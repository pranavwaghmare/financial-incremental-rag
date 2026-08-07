from app.database.chroma_db import ChromaDBClient
from app.services.embedding_service import EmbeddingService


class IndexService:

    def __init__(self):

        self.chroma = ChromaDBClient()

        self.collection = self.chroma.collection

        self.embedder = EmbeddingService()

    def add_chunk(
        self,
        chunk_id,
        text,
        metadata
    ):

        embedding = self.embedder.generate_embedding(text)

        self.collection.add(

            ids=[chunk_id],

            documents=[text],

            embeddings=[embedding],

            metadatas=[metadata]

        )

    def update_chunk(
        self,
        chunk_id,
        text,
        metadata
    ):

        self.collection.delete(ids=[chunk_id])

        embedding = self.embedder.generate_embedding(text)

        self.collection.add(

            ids=[chunk_id],

            documents=[text],

            embeddings=[embedding],

            metadatas=[metadata]

        )

    def delete_chunk(self, chunk_id):

        self.collection.delete(

            ids=[chunk_id]

        )