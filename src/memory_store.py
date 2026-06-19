from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path


PROFILE_FIELDS = (
    "name",
    "location",
    "profession",
    "style",
    "drink",
    "food",
    "pet",
    "interests",
    "notes",
)


def estimate_tokens(text: str) -> int:
    """Approximate token count with a stable heuristic."""

    cleaned = text.strip()
    if not cleaned:
        return 0
    words = re.findall(r"\S+", cleaned)
    return max(1, len(cleaned) // 4 + len(words) // 2)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = slug.strip("._")
    return slug or "user"


def _default_profile_text() -> str:
    lines = ["# User Profile", ""]
    for field in PROFILE_FIELDS:
        lines.append(f"- {field}:")
    return "\n".join(lines) + "\n"


def _parse_profile_text(content: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    for line in content.splitlines():
        match = re.match(r"^\s*-\s*([A-Za-z_]+)\s*:\s*(.*)\s*$", line)
        if not match:
            continue
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        if key in PROFILE_FIELDS and value:
            facts[key] = value
    return facts


def _render_profile_text(facts: dict[str, str]) -> str:
    lines = ["# User Profile", ""]
    for field in PROFILE_FIELDS:
        value = facts.get(field, "").strip()
        lines.append(f"- {field}: {value}")
    return "\n".join(lines) + "\n"


def _normalize_value(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" \t\r\n,;:.!?")
    return cleaned


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / f"{_slugify(user_id)}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return _default_profile_text()
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(user_id)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        path = self.path_for(user_id)
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        updated = content.replace(search_text, replacement, 1)
        path.write_text(updated, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        return _parse_profile_text(self.read_text(user_id))

    def upsert_fact(self, user_id: str, field: str, value: str) -> Path:
        field = field.strip().lower()
        if field not in PROFILE_FIELDS:
            raise ValueError(f"Unsupported profile field: {field}")
        facts = self.facts(user_id)
        facts[field] = _normalize_value(value)
        return self.write_text(user_id, _render_profile_text(facts))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract stable profile facts from a user message."""

    text = message.strip()
    if not text:
        return {}

    if "?" in text and not re.search(
        r"\b(tên|ở|làm|thích|nuôi|đồ uống|món ăn|style|trả lời|đính chính|giờ|hiện tại|mình là)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return {}

    updates: dict[str, str] = {}

    def add(field: str, value: str | None) -> None:
        if not value:
            return
        cleaned = _normalize_value(value)
        if cleaned:
            updates[field] = cleaned

    patterns = [
        ("name", [
            r"\b(?:mình|tôi)\s+tên\s+là\s+([^.;!?]+)",
            r"\btên\s+mình\s+là\s+([^.;!?]+)",
            r"\bmình\s+là\s+([^.;!?]+)",
        ]),
        ("location", [
            r"\b(?:hiện\s+tại\s+)?(?:mình\s+)?đang\s+ở\s+([^.;!?]+)",
            r"\b(?:mình\s+)?ở\s+([^.;!?]+)",
            r"\b(?:nơi\s+ở|địa\s+điểm)\s+hiện\s+tại\s+là\s+([^.;!?]+)",
        ]),
        ("profession", [
            r"\b(?:mình\s+)?(?:đang\s+)?làm\s+([^.;!?]+?)(?:\s+cho\s+[^.;!?]+)?(?:$|[.,;!?])",
            r"\b(?:giờ\s+)?(?:chuyển\s+sang|làm\s+nghề|nghề\s+nghiệp\s+là)\s+([^.;!?]+)",
            r"\b(?:không\s+còn\s+làm\s+[^,.;!?]+,\s*)?(?:giờ\s+)?(?:mình\s+)?là\s+([^.;!?]+)",
        ]),
        ("drink", [
            r"\bđồ\s+uống\s+yêu\s+thích\s+là\s+([^.;!?]+)",
            r"\bvẫn\s+uống\s+([^.;!?]+)",
        ]),
        ("food", [
            r"\bmón\s+ăn\s+yêu\s+thích\s+là\s+([^.;!?]+)",
        ]),
        ("pet", [
            r"\bnuôi\s+(?:một\s+bé\s+)?([^.;!?]+)",
            r"\bcon\s+([^.;!?]+)\s+tên\s+([^.;!?]+)",
        ]),
    ]

    for field, field_patterns in patterns:
        for pattern in field_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            if field == "pet" and match.lastindex == 2:
                add(field, f"{match.group(1)} tên {match.group(2)}")
            else:
                add(field, match.group(1))
            break

    style_parts: list[str] = []
    style_checks = [
        ("ngắn gọn", "ngắn gọn"),
        ("bullet", "bullet"),
        ("ví dụ thực chiến", "ví dụ thực chiến"),
        ("ví dụ thực tế", "ví dụ thực tế"),
        ("rõ ý", "rõ ý"),
        ("3 bullet", "3 bullet"),
        ("có cấu trúc", "có cấu trúc"),
        ("trade-off", "trade-off"),
    ]
    for needle, label in style_checks:
        if re.search(re.escape(needle), text, flags=re.IGNORECASE):
            style_parts.append(label)
    if style_parts:
        add("style", ", ".join(dict.fromkeys(style_parts)))

    interests_parts: list[str] = []
    if re.search(r"\bPython\b", text, flags=re.IGNORECASE):
        interests_parts.append("Python")
    if re.search(r"\bAI ứng dụng\b", text, flags=re.IGNORECASE):
        interests_parts.append("AI ứng dụng")
    if re.search(r"\bMLOps\b", text, flags=re.IGNORECASE):
        interests_parts.append("MLOps")
    if re.search(r"\bbenchmark\b", text, flags=re.IGNORECASE):
        interests_parts.append("benchmark")
    if interests_parts:
        add("interests", ", ".join(dict.fromkeys(interests_parts)))

    if "đính chính" in text.lower() or "ưu tiên" in text.lower() or "nhớ" in text.lower():
        add("notes", _normalize_value(text[:180]))

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary from older messages."""

    if not messages:
        return ""
    tail = messages[-max_items:]
    parts: list[str] = []
    for message in tail:
        role = message.get("role", "unknown")
        content = _normalize_value(message.get("content", ""))
        if len(content) > 120:
            content = content[:117].rstrip() + "..."
        parts.append(f"{role}: {content}")
    return " | ".join(parts)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread_state(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )

    def _context_text(self, thread_id: str) -> str:
        state = self._thread_state(thread_id)
        chunks: list[str] = []
        summary = str(state.get("summary", "")).strip()
        if summary:
            chunks.append(summary)
        for message in state.get("messages", []):
            if isinstance(message, dict):
                chunks.append(f"{message.get('role', 'unknown')}: {message.get('content', '')}")
        return "\n".join(chunks)

    def _compact_if_needed(self, thread_id: str) -> None:
        state = self._thread_state(thread_id)
        keep = max(1, int(self.keep_messages))
        while True:
            messages = state.get("messages", [])
            if not isinstance(messages, list):
                messages = []
                state["messages"] = messages
            current_tokens = estimate_tokens(self._context_text(thread_id))
            if current_tokens <= self.threshold_tokens or len(messages) <= keep:
                break
            older = messages[:-keep]
            recent = messages[-keep:]
            existing_summary = str(state.get("summary", "")).strip()
            new_summary = summarize_messages(older, max_items=min(len(older), 6))
            summary_parts = [part for part in [existing_summary, new_summary] if part]
            merged = " || ".join(summary_parts)
            if len(merged) > 1000:
                merged = merged[:997].rstrip() + "..."
            state["summary"] = merged
            state["messages"] = recent
            state["compactions"] = int(state.get("compactions", 0)) + 1

    def append(self, thread_id: str, role: str, content: str) -> None:
        state = self._thread_state(thread_id)
        messages = state.setdefault("messages", [])
        if not isinstance(messages, list):
            messages = []
            state["messages"] = messages
        messages.append({"role": role, "content": content})
        self._compact_if_needed(thread_id)

    def context(self, thread_id: str) -> dict[str, object]:
        state = self._thread_state(thread_id)
        messages = state.get("messages", [])
        if isinstance(messages, list):
            copied_messages = [dict(message) for message in messages if isinstance(message, dict)]
        else:
            copied_messages = []
        return {
            "summary": str(state.get("summary", "")),
            "messages": copied_messages,
            "compactions": int(state.get("compactions", 0)),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self._thread_state(thread_id).get("compactions", 0))
