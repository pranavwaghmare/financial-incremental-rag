from pathlib import Path
import chromadb

CHROMA_PATH = "chroma_storage"
COLLECTION_NAME = "financial_documents"

_client = None
_collection = None


def initialize_chromadb():
    global _client, _collection

    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME
        )
        print("ChromaDB Initialized")


def get_collection():
    global _collection

    if _collection is None:
        initialize_chromadb()

    return _collection


class ChromaDBClient:

    def __init__(self):
        self.collection = get_collection()

    # -----------------------------
    # Bulk insert (Milestone 2)
    # -----------------------------
    def add_chunks(self, document_name: str, chunks: list, embeddings):

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:

            chunk_id = f"{document_name}_{chunk['chunk_number']}"

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

        self.collection.add(
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