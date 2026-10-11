import hashlib
from collections import defaultdict
from typing import List


class HashingService:
    """
    Generates SHA-256 hashes for text chunks.
    """

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text before hashing.

        Removes extra spaces and line breaks so that
        formatting changes do not produce different hashes.
        """

        return " ".join(text.split()).strip()

    @staticmethod
    def generate_hash(text: str) -> str:
        """
        Generate SHA-256 hash for a single chunk.
        """

        normalized_text = HashingService.normalize_text(text)

        return hashlib.sha256(
            normalized_text.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def generate_hashes(chunks: List[str]) -> List[str]:
        """
        Generate SHA-256 hashes for multiple chunks.
        """

        return [
            HashingService.generate_hash(chunk)
            for chunk in chunks
        ]

    @staticmethod
    def chunk_id(document_name: str, chunk_hash: str) -> str:
        """Build a stable vector and manifest ID from document and content."""
        return f"{document_name}_{chunk_hash}"

    @staticmethod
    def generate_chunk_ids(document_name: str, hashes: List[str]) -> List[str]:
        """Build stable IDs and disambiguate repeated identical chunks."""
        occurrences = defaultdict(int)
        chunk_ids = []

        for chunk_hash in hashes:
            occurrence = occurrences[chunk_hash]
            occurrences[chunk_hash] += 1
            suffix = f"_{occurrence}" if occurrence else ""
            chunk_ids.append(f"{document_name}_{chunk_hash}{suffix}")

        return chunk_ids