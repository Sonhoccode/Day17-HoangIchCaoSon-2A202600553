from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model
from agent_baseline import _answer_question, _summarize_facts


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Advanced Agent with short-term, persistent, and compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

    def _thread_tokens(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def _thread_prompt_tokens(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._thread_tokens(thread_id)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._thread_prompt_tokens(thread_id)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        updates = extract_profile_updates(message)
        for field, value in updates.items():
            self.profile_store.upsert_fact(user_id, field, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self._thread_prompt_tokens(thread_id) + prompt_tokens

        reply = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", reply)
        self.thread_tokens[thread_id] = self._thread_tokens(thread_id) + estimate_tokens(reply)

        return {
            "reply": reply,
            "thread_id": thread_id,
            "user_id": user_id,
            "token_usage": self.thread_tokens[thread_id],
            "prompt_tokens_processed": self.thread_prompt_tokens[thread_id],
            "memory_file_size": self.memory_file_size(user_id),
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_text = self.profile_store.read_text(user_id)
        context = self.compact_memory.context(thread_id)
        parts: list[str] = [profile_text]
        summary = str(context.get("summary", "")).strip()
        if summary:
            parts.append(summary)
        for message in context.get("messages", []):
            if isinstance(message, dict):
                parts.append(f"{message.get('role', 'unknown')}: {message.get('content', '')}")
        return estimate_tokens("\n".join(parts))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        context = self.compact_memory.context(thread_id)
        thread_messages = context.get("messages", [])
        if not isinstance(thread_messages, list):
            thread_messages = []
        answer = _answer_question(message, facts, thread_messages)  # type: ignore[arg-type]

        if answer.startswith("Mình chưa") and facts:
            summary = _summarize_facts(facts)
            if summary:
                answer = f"Mình đang nhớ: {summary}."
        return answer

    def _maybe_build_langchain_agent(self):
        if self.langchain_agent is None:
            try:
                self.langchain_agent = build_chat_model(self.config.model)
            except Exception:
                self.langchain_agent = None
        return self.langchain_agent
