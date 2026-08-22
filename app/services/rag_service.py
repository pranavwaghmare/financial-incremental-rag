from app.services.retrieval_service import RetrievalService
from app.services.groq_service import GroqService
from app.services.prompt_builder import build_prompt
import time

from app.services.analytics_service import AnalyticsService

class RAGService:

    def __init__(
        self,
        retriever: RetrievalService = None,
        groq_service: GroqService = None,
        analytics_service: AnalyticsService = None,
    ):
        self.retriever = retriever or RetrievalService()
        self.groq = groq_service or GroqService()
        self.analytics = analytics_service or AnalyticsService()

    def ask(self, question: str, top_k: int = 5, log_analytics: bool = True):
        start = time.perf_counter()

        # Step 1
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            question=question,
            top_k=top_k
        )

        # No chunks found
        if len(chunks) == 0:

            return {

                "answer": "No relevant information found.",

                "sources": []

            }

        # Step 2
        # Build context

        context = ""

        for chunk in chunks:

            context += f"""

Document:
{chunk['document']}

Page:
{chunk['page']}

Content:
{chunk['text']}

"""

        # Step 3
        # Build prompt

        prompt = build_prompt(
            question=question,
            context=context
        )

        # Step 4
        # Generate answer

        answer = self.groq.generate_answer(prompt)

        end = time.perf_counter()

        response_time = round(end - start, 3)

        # Save query analytics
        if log_analytics:
            self.analytics.save_query_metrics(
                question=question,
                response_time=response_time,
                document=chunks[0]["document"],
                page=chunks[0]["page"]
            )

        return {

            "answer": answer,

            "sources": chunks

        }