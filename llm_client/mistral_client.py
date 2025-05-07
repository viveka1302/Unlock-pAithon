# llm_client/mistral_client.py
from .base import LLMClient
import requests

class MistralClient(LLMClient):
    def __init__(self, api_key: str, model: str = "mistral-small"):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7)
        }
        response = requests.post("https://api.mistral.ai/v1/chat/completions", json=payload, headers=headers)
        return response.json()["choices"][0]["message"]["content"].strip()
