import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class OpenRouterClient:

    def __init__(
        self,
        model: str = "openrouter/free",
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set"
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        self.model = model

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.2,
    ):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )

        return response.choices[0].message.content