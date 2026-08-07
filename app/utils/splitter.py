from langchain_text_splitters import RecursiveCharacterTextSplitter


class FinancialTextSplitter:

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def split_pages(self, pages):

        chunks = []

        chunk_number = 0

        for page in pages:

            page_chunks = self.splitter.split_text(page["text"])

            for chunk in page_chunks:
                chunk_number += 1

                chunks.append(
                    {
                        "chunk_number": chunk_number,
                        "page": page["page"],
                        "text": chunk
                    }
                )

        return chunks