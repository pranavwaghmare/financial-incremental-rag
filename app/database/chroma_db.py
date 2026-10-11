from pathlib import Path
import chromadb

from app.services.hashing_service import HashingService

CHROMA_PATH = "chroma_storage"
COLLECTION_NAME = "financial_documents"

_client = None
_collection = None


def initialize_chromadb(chroma_path: str = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
    global _client, _collection

    Path(chroma_path).mkdir(parents=True, exist_ok=True)

    if _client is None or getattr(_client, "_path", None) != chroma_path or getattr(_collection, "name", None) != collection_name:
        _client = chromadb.PersistentClient(path=chroma_path)
        _collection = _client.get_or_create_collection(
            name=collection_name
        )
        print(f"ChromaDB Initialized at {chroma_path} ({collection_name})")


def get_collection():
    global _collection

    if _collection is None:
        initialize_chromadb()

    return _collection


class ChromaDBClient:

    def __init__(self, path: str = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
        self.path = path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    # -----------------------------
    # Bulk insert (Milestone 2)
    # -----------------------------
    def add_chunks(self, document_name: str, chunks: list, embeddings):

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:

            chunk_id = HashingService.chunk_id(
                document_name,
                HashingService.generate_hash(chunk["text"]),
            )

            ids.append(chunk_id)
            documents.append(chunk["text"])

            metadatas.append(
                {
                    "document_name": document_name,
                    "page": int(chunk["page"]),
                    "chunk_number": int(chunk["chunk_number"]),
                }
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # -----------------------------
    # Incremental Add
    # -----------------------------
    def add_chunk(
        self,
        chunk_id,
        text,
        embedding,
        metadata
    ):

        self.collection.upsert(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    # -----------------------------
    # Incremental Update
    # -----------------------------
    def update_chunk(
        self,
        chunk_id,
        text,
        embedding,
        metadata
    ):

        self.collection.upsert(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    # -----------------------------
    # Incremental Delete
    # -----------------------------
    def delete_chunk(self, chunk_id):

        self.collection.delete(
            ids=[chunk_id]
        )

    # -----------------------------
    # Stats
    # -----------------------------
    def collection_count(self):

        return self.collection.count()


# --------------------------------
# Convenience functions
# --------------------------------

def add_chunks(document_name, chunks, embeddings):

    client = ChromaDBClient()

    client.add_chunks(
        document_name,
        chunks,
        embeddings
    )


def collection_count():

    client = ChromaDBClient()

    return client.collection_count()