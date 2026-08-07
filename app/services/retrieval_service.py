from app.database.chroma_db import get_collection
from app.services.embedding_service import EmbeddingService


class RetrievalService:

    def __init__(self):
        self.embedder = EmbeddingService()
        self.get_collection = get_collection

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        collection = self.get_collection()

        # Handle empty collection safely
        if collection.count() == 0:
            return []

        # Call method on EmbeddingService class instance
        query_embedding = self.embedder.embed_query(question)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved_chunks = []
        for i in range(len(ids)):
            retrieved_chunks.append(
                {
                    "chunk_id": ids[i],
                    "text": docs[i] if i < len(docs) else "",
                    "page": (
                        metas[i].get("page", "Unknown")
                        if i < len(metas) and metas[i]
                        else "Unknown"
                    ),
                    "document": (
                        metas[i].get("document_name", "Unknown")
                        if i < len(metas) and metas[i]
                        else "Unknown"
                    ),
                    "distance": distances[i] if i < len(distances) else None,
                }
            )

        return retrieved_chunks