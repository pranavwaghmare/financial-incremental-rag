import sqlite3
from pathlib import Path


class ManifestDB:

    def __init__(self, db_path: str = "manifest_db/manifest.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.create_tables()

    def get_cursor(self):
        """Helper to return a fresh cursor per thread operation."""
        return self.conn.cursor()

    def create_tables(self):
        cursor = self.get_cursor()

        # 1. Document-level manifest (for DocumentProcessingService compatibility)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                page_count INTEGER,
                chunk_count INTEGER,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT
            )
            """
        )

        # 2. Chunk-level manifest (for Incremental Hash tracking)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS manifest (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_name TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL UNIQUE,
                chunk_index INTEGER,
                page INTEGER,
                chunk_hash TEXT NOT NULL,
                embedding_id TEXT,
                version INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.conn.commit()

    # --- Document-Level Methods ---
    def insert_document(self, filename: str, page_count: int, chunk_count: int):
        cursor = self.get_cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO documents
            (filename, page_count, chunk_count, status)
            VALUES (?, ?, ?, ?)
            """,
            (filename, page_count, chunk_count, "Indexed"),
        )
        self.conn.commit()

    # --- Chunk-Level Incremental RAG Methods ---
    def insert_chunk(
        self,
        document_name: str,
        document_id: str,
        chunk_id: str,
        chunk_index: int,
        page: int,
        chunk_hash: str,
        embedding_id: str = None,
        version: int = 1,
    ):
        cursor = self.get_cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO manifest
            (document_name, document_id, chunk_id, chunk_index, page, chunk_hash, embedding_id, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_name,
                document_id,
                chunk_id,
                chunk_index,
                page,
                chunk_hash,
                embedding_id,
                version,
            ),
        )
        self.conn.commit()

    def chunk_exists(self, chunk_hash: str) -> bool:
        """Check if a chunk hash already exists in DB to prevent re-embedding."""
        cursor = self.get_cursor()
        cursor.execute(
            "SELECT 1 FROM manifest WHERE chunk_hash = ?", (chunk_hash,)
        )
        return cursor.fetchone() is not None