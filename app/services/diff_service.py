from typing import Dict, List


class DiffService:
    """
    Compares chunk hashes between
    an old document and a new document.
    """

    @staticmethod
    def compare(
        old_hashes: Dict[int, str],
        new_hashes: Dict[int, str]
    ) -> dict:

        new_chunks = []
        modified_chunks = []
        deleted_chunks = []
        unchanged_chunks = []

        old_indexes = set(old_hashes.keys())
        new_indexes = set(new_hashes.keys())

        # Existing chunk indexes
        common_indexes = old_indexes.intersection(new_indexes)

        for index in common_indexes:

            if old_hashes[index] == new_hashes[index]:

                unchanged_chunks.append(index)

            else:

                modified_chunks.append(index)

        # New chunk indexes

        for index in new_indexes - old_indexes:

            new_chunks.append(index)

        # Deleted chunk indexes

        for index in old_indexes - new_indexes:

            deleted_chunks.append(index)

        return {

            "new": new_chunks,

            "modified": modified_chunks,

            "deleted": deleted_chunks,

            "unchanged": unchanged_chunks

        }