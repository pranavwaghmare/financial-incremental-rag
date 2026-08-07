def build_prompt(question: str, context: str) -> str:
    """
    Builds the RAG prompt sent to Groq.
    """

    prompt = f"""
You are an expert financial analyst.

Use ONLY the context below to answer the question.

If the answer cannot be found in the context,
reply exactly:

I could not find this information in the uploaded financial documents.

Context
-------
{context}

Question
--------
{question}

Answer
------
"""

    return prompt