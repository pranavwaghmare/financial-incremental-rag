import re


class TextCleaner:

    @staticmethod
    def clean(text: str) -> str:
        """Basic cleaning for financial documents."""

        if not text:
            return ""

        # Replace multiple spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Replace multiple newlines
        text = re.sub(r"\n+", "\n", text)

        # Remove leading/trailing spaces
        text = text.strip()

        return text