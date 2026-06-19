from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


def _merge_facts_from_messages(messages: list[dict[str, str]]) -> dict[str, str]:
    facts: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "user":
            continue
        facts.update(extract_profile_updates(message.get("content", "")))
    return facts


def _summarize_facts(facts: dict[str, str]) -> str:
    parts: list[str] = []
    for key in ("name", "profession", "location", "style", "drink", "food", "pet", "interests"):
        value = facts.get(key)
        if value:
            parts.append(f"{key}={value}")
    return "; ".join(parts)


def _answer_question(message: str, facts: dict[str, str], thread_messages: list[dict[str, str]]) -> str:
    text = message.strip().lower()
    name = facts.get("name", "")
    location = facts.get("location", "")
    profession = facts.get("profession", "")
    style = facts.get("style", "")
    drink = facts.get("drink", "")
    food = facts.get("food", "")
    pet = facts.get("pet", "")
    interests = facts.get("interests", "")

    if re.search(r"\b(tên|name)\b", text) and not re.search(r"\bstyle\b", text):
        if name:
            return f"Tên bạn là {name}."
        return "Mình chưa thấy bạn giới thiệu tên trong thread này."

    if re.search(r"\b(ở đâu|đang ở đâu|nơi ở|location|hiện ở)\b", text):
        if location:
            return f"Hiện tại bạn đang ở {location}."
        return "Mình chưa thấy thông tin nơi ở trong thread này."

    if re.search(r"\b(nghề|làm nghề gì|đang làm gì|profession|công việc)\b", text):
        if profession:
            return f"Hiện tại bạn đang làm {profession}."
        return "Mình chưa thấy thông tin nghề nghiệp trong thread này."

    if re.search(r"\b(style|trả lời|ngắn gọn|bullet|ví dụ)\b", text):
        if style:
            return f"Style bạn thích là {style}."
        return "Mình chưa thấy preference về style trong thread này."

    if re.search(r"\b(đồ uống|uống gì|drink)\b", text):
        if drink:
            return f"Đồ uống yêu thích của bạn là {drink}."
        return "Mình chưa thấy thông tin đồ uống yêu thích trong thread này."

    if re.search(r"\b(món ăn|ăn gì|food)\b", text):
        if food:
            return f"Món ăn yêu thích của bạn là {food}."
        return "Mình chưa thấy thông tin món ăn yêu thích trong thread này."

    if re.search(r"\b(con gì|nuôi|pet|corgi)\b", text):
        if pet:
            return f"Bạn nuôi {pet}."
        return "Mình chưa thấy thông tin thú cưng trong thread này."

    if re.search(r"\b(quan tâm|thích|interest|mối quan tâm)\b", text):
        if interests:
            return f"Mối quan tâm chính của bạn là {interests}."
        return "Mình chưa thấy thông tin mối quan tâm trong thread này."

    if re.search(r"\b(tóm tắt|mô tả|hãy nói về mình|bạn là ai|mình là ai)\b", text):
        summary = _summarize_facts(facts)
        if summary:
            return f"Trong thread này mình nhớ: {summary}."
        return "Mình chưa có đủ thông tin để mô tả bạn trong thread này."

    if thread_messages:
        recent = thread_messages[-2:]
        recent_text = " | ".join(m.get("content", "") for m in recent)
        if len(recent_text) > 140:
            recent_text = recent_text[:137].rstrip() + "..."
        return f"Mình đã ghi nhận: {recent_text}"

    return "Mình chưa có đủ ngữ cảnh để trả lời rõ hơn."


class BaselineAgent:
    """Baseline Agent with within-session memory only."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        state = self._session(thread_id)
        prompt_text = "\n".join(item["content"] for item in state.messages)
        prompt_tokens = estimate_tokens(prompt_text + ("\n" if prompt_text else "") + message)
        state.prompt_tokens_processed += prompt_tokens

        state.messages.append({"role": "user", "content": message})
        facts = _merge_facts_from_messages(state.messages)
        reply = _answer_question(message, facts, state.messages)
        state.messages.append({"role": "assistant", "content": reply})
        state.token_usage += estimate_tokens(reply)

        return {
            "reply": reply,
            "thread_id": thread_id,
            "token_usage": state.token_usage,
            "prompt_tokens_processed": state.prompt_tokens_processed,
        }

    def _maybe_build_langchain_agent(self):
        if self.langchain_agent is None:
            try:
                self.langchain_agent = build_chat_model(self.config.model)
            except Exception:
                self.langchain_agent = None
        return self.langchain_agent
