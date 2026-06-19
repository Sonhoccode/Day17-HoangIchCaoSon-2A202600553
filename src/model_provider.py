from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider settings shared by the agents."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map provider aliases to supported provider names."""

    normalized = value.strip().lower()
    aliases = {
        "anthorpic": "anthropic",
        "anthropic": "anthropic",
        "gpt": "openai",
        "openai": "openai",
        "custom": "custom",
        "gemini": "gemini",
        "ollama": "ollama",
        "openrouter": "openrouter",
    }
    return aliases.get(normalized, normalized)


def build_chat_model(config: ProviderConfig):
    """Instantiate the chat model for the selected provider.

    Imports are deferred so this module stays importable in offline test runs
    even if provider SDKs are missing.
    """

    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            anthropic_api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs: dict[str, object] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    raise ValueError(f"Unsupported provider: {provider}")
