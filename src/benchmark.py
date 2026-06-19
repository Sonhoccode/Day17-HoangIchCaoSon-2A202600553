from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import estimate_tokens


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Load JSON conversations from disk."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Score 0 / 0.5 / 1 based on matched expected facts."""

    if not expected:
        return 0.0
    hits = sum(1 for item in expected if item.lower() in answer.lower())
    if hits == len(expected):
        return 1.0
    if hits > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight offline quality heuristic."""

    recall = recall_points(answer, expected)
    tokens = estimate_tokens(answer)
    brevity = 1.0 if tokens <= 40 else max(0.0, 1.0 - min(tokens, 120) / 120.0)
    structure_bonus = 0.1 if "- " in answer or "•" in answer or ":" in answer else 0.0
    score = min(1.0, recall * 0.7 + brevity * 0.2 + structure_bonus)
    return round(score, 3)


def _memory_size(agent, user_ids: set[str]) -> int:
    if not user_ids:
        return 0
    if not hasattr(agent, "memory_file_size"):
        return 0
    return sum(int(agent.memory_file_size(user_id)) for user_id in sorted(user_ids))


def _compaction_total(agent, thread_ids: set[str]) -> int:
    if not hasattr(agent, "compaction_count"):
        return 0
    return sum(int(agent.compaction_count(thread_id)) for thread_id in sorted(thread_ids))


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations."""

    user_ids: set[str] = set()
    thread_ids: set[str] = set()
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conversation in conversations:
        user_id = conversation["user_id"]
        thread_id = conversation["id"]
        user_ids.add(user_id)
        thread_ids.add(thread_id)

        for turn in conversation.get("turns", []):
            agent.reply(user_id=user_id, thread_id=thread_id, message=turn)

        for index, question in enumerate(conversation.get("recall_questions", []), start=1):
            recall_thread_id = f"{thread_id}-recall-{index}"
            thread_ids.add(recall_thread_id)
            result = agent.reply(user_id=user_id, thread_id=recall_thread_id, message=question["question"])
            answer = result["reply"]
            expected = question.get("expected_contains", [])
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    total_agent_tokens = sum(int(agent.token_usage(thread_id)) for thread_id in thread_ids if hasattr(agent, "token_usage"))
    total_prompt_tokens = sum(int(agent.prompt_token_usage(thread_id)) for thread_id in thread_ids if hasattr(agent, "prompt_token_usage"))

    if isinstance(agent, BaselineAgent):
        total_agent_tokens = sum(agent.token_usage(thread_id) for thread_id in thread_ids)
        total_prompt_tokens = sum(agent.prompt_token_usage(thread_id) for thread_id in thread_ids)

    recall_score = round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0
    response_quality = round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=recall_score,
        response_quality=response_quality,
        memory_growth_bytes=_memory_size(agent, user_ids),
        compactions=_compaction_total(agent, thread_ids),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render rows as a markdown table."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    data = [
        [
            row.agent_name,
            str(row.agent_tokens_only),
            str(row.prompt_tokens_processed),
            f"{row.recall_score:.3f}",
            f"{row.response_quality:.3f}",
            str(row.memory_growth_bytes),
            str(row.compactions),
        ]
        for row in rows
    ]

    try:
        from tabulate import tabulate

        return tabulate(data, headers=headers, tablefmt="github")
    except Exception:
        widths = [len(h) for h in headers]
        for row in data:
            for index, value in enumerate(row):
                widths[index] = max(widths[index], len(value))
        lines = []
        lines.append(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
        lines.append("-|-".join("-" * widths[i] for i in range(len(headers))))
        for row in data:
            lines.append(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
        return "\n".join(lines)


def _print_section(title: str, rows: list[BenchmarkRow]) -> None:
    print(title)
    print(format_rows(rows))
    print()


def main() -> None:
    """Run the standard and long-context benchmarks."""

    config = load_config(Path(__file__).resolve().parent.parent)
    standard = load_conversations(config.data_dir / "conversations.json")
    stress = load_conversations(config.data_dir / "advanced_long_context.json")

    baseline_standard = BaselineAgent(config=config, force_offline=True)
    advanced_standard = AdvancedAgent(config=config, force_offline=True)
    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    standard_rows = [
        run_agent_benchmark("Baseline", baseline_standard, standard, config),
        run_agent_benchmark("Advanced", advanced_standard, standard, config),
    ]
    stress_rows = [
        run_agent_benchmark("Baseline", baseline_stress, stress, config),
        run_agent_benchmark("Advanced", advanced_stress, stress, config),
    ]

    _print_section("Standard Benchmark", standard_rows)
    _print_section("Long-Context Stress Benchmark", stress_rows)


if __name__ == "__main__":
    main()
