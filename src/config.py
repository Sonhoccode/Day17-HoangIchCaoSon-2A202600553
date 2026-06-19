from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def normalize_provider(value: str) -> str:
    """Normalize provider aliases to the supported provider names."""

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


def _load_env_file(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv
    except Exception:
        return

    load_dotenv(env_path, override=False)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _provider_env(provider: str) -> dict[str, str | None]:
    provider = normalize_provider(provider)
    if provider == "openai":
        return {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL"),
        }
    if provider == "custom":
        return {
            "api_key": os.getenv("CUSTOM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("CUSTOM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        }
    if provider == "gemini":
        return {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "base_url": None,
        }
    if provider == "anthropic":
        return {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "base_url": None,
        }
    if provider == "ollama":
        return {
            "api_key": os.getenv("OLLAMA_API_KEY"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        }
    if provider == "openrouter":
        return {
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        }
    raise ValueError(f"Unsupported provider: {provider}")


def _default_model_name(provider: str) -> str:
    provider = normalize_provider(provider)
    defaults = {
        "openai": "gpt-4o-mini",
        "custom": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-5-sonnet-latest",
        "ollama": "llama3.1",
        "openrouter": "openai/gpt-4o-mini",
    }
    if provider not in defaults:
        raise ValueError(f"Unsupported provider: {provider}")
    return defaults[provider]


def _provider_config_from_env(prefix: str, fallback_provider: str) -> ProviderConfig:
    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", fallback_provider))
    env = _provider_env(provider)

    model_name = os.getenv(f"{prefix}_MODEL", _default_model_name(provider))
    temperature = _env_float(f"{prefix}_TEMPERATURE", 0.2 if prefix == "LLM" else 0.0)

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=env["api_key"],
        base_url=env["base_url"],
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a populated lab config."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_env_file(root)

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    fallback_provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    compact_threshold_tokens = _env_int("COMPACT_THRESHOLD_TOKENS", 2500)
    compact_keep_messages = _env_int("COMPACT_KEEP_MESSAGES", 8)

    model = _provider_config_from_env("LLM", fallback_provider)
    judge_model = _provider_config_from_env("JUDGE", fallback_provider)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model,
        judge_model=judge_model,
    )


def build_chat_model(config: ProviderConfig):
    """Instantiate a chat model for the selected provider.

    Imports are deferred so the module can still be imported without the
    provider-specific packages installed.
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
