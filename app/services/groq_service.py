import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

# Initialize client once
client = Groq(api_key=GROQ_API_KEY)


class GroqService:
    """
    Handles all communication with the Groq API.
    """

    def __init__(self):
        self.client = client

    def generate_answer(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial analyst. "
                        "Answer only using the provided context. "
                        "If the answer is not present, clearly say so."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_completion_tokens=1024
        )

        return response.choices[0].message.content