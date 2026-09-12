from app.llm.client import OpenRouterClient


llm = OpenRouterClient()


response = llm.generate(
    [
        {
            "role": "user",
            "content": "Explain RAG in one sentence.",
        }
    ]
)


print("\nLLM RESPONSE:\n")
print(response)