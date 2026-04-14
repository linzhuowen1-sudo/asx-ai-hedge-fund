"""LLM provider abstraction — supports OpenAI, Anthropic, Ollama, etc."""

import os
from typing import Optional

from langchain_core.language_models import BaseChatModel


def get_llm(
    model_name: Optional[str] = None,
    model_provider: Optional[str] = None,
    temperature: float = 0,
) -> BaseChatModel:
    """Get an LLM instance based on provider and model name.

    Defaults to environment variables:
        LLM_PROVIDER (default: "openai")
        LLM_MODEL (default: "gpt-4o")
    """
    provider = (model_provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    model = model_name or os.getenv("LLM_MODEL", "qwen3.5:27b-nvfp4")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature)

    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature)

    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=temperature,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
