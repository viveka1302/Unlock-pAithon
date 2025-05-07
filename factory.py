
import os
from llm_client.openai_client import OpenAIClient
from llm_client.anthropic_client import AnthropicClient
from llm_client.mistral_client import MistralClient
from llm_client.llama_client import llamaClient

def get_llm_client(provider: str):
    provider = provider.lower()
    if provider == "openai":
        return OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "anthropic":
        return AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif provider == "mistral":
        return MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
    elif provider == "ollama":
        return OllamaClient()
    else:
        raise ValueError(f"Unsupported provider: {provider}")
