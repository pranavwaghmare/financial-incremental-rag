from typing import Dict


class DiffService:
    """
    Compares chunk hashes between an old document and a new document using
    content-addressable set membership instead of positional index comparison.

    A chunk is unchanged if its hash value exists anywhere in the previous
    version, regardless of its current index in the document.
    """

    @staticmethod
    def compare(
        old_hashes: Dict[int, str],
        new_hashes: Dict[int, str]
    ) -> dict:

        old_hash_values = set(old_hashes.values())
        new_hash_values = set(new_hashes.values())

        new_chunks = []
        modified_chunks = []
        deleted_chunks = []
        unchanged_chunks = []

        # A hash is unchanged when it existed anywhere in the previous version.
        for index, hash_value in new_hashes.items():
            if hash_value in old_hash_values:
                unchanged_chunks.append(index)
            else:
                new_chunks.append(index)

        # Any old hash that is no longer present in the new document is deleted.
        for index, hash_value in old_hashes.items():
            if hash_value not in new_hash_values:
                deleted_chunks.append(index)

        # With pure set-based hashing, a modified chunk is indistinguishable from a
        # removed hash plus a new hash, so "modified" remains empty by design.
        return {
            "new": new_chunks,
            "modified": modified_chunks,
            "deleted": deleted_chunks,
            "unchanged": unchanged_chunks,
        }