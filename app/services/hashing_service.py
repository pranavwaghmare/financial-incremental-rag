import hashlib
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