from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingService:

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate 2D vector embeddings for a list of document chunks."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_text(
        self,
        text: str
    ):

        embedding = self.model.encode(

            text,

            convert_to_numpy=True

        )

        return embedding.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate a single 1D vector embedding for a query string."""
        if isinstance(query, list):
            query = query[0]
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()