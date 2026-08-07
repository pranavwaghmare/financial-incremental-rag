import pymupdf

from pathlib import Path


class PDFLoader:
    def extract_text(self, file_path: str):
        """Extract text from PDF page by page."""

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc = pymupdf.open(file_path)

        pages = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")

            pages.append(
                {
                    "page": page_num + 1,
                    "text": text
                }
            )

        doc.close()

        return pages