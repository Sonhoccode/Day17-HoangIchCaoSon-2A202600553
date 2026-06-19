from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path):
    """Build an isolated config for tests."""

    root = Path(__file__).resolve().parent.parent
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=120,
        compact_keep_messages=4,
        model=ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")

    path = store.write_text("dungct", "# User Profile\n\n- name: DũngCT\n")
    assert path.exists()
    assert "DũngCT" in store.read_text("dungct")

    assert store.edit_text("dungct", "DũngCT", "DũngCT Pro")
    assert "DũngCT Pro" in store.read_text("dungct")


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    for index in range(12):
        agent.reply(
            user_id="dungct",
            thread_id="thread-compact",
            message=f"Turn {index}: Mình đang ở Huế và thích trả lời ngắn gọn, có ví dụ thực chiến. {index} " + ("x" * 80),
        )

    assert agent.compaction_count("thread-compact") > 0
    assert len(agent.compact_memory.context("thread-compact")["messages"]) <= config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    baseline.reply("dungct", "thread-a", "Mình tên là DũngCT, mình đang ở Huế và làm MLOps engineer.")
    advanced.reply("dungct", "thread-a", "Mình tên là DũngCT, mình đang ở Huế và làm MLOps engineer.")

    baseline_answer = baseline.reply("dungct", "thread-b", "Mình tên gì?")["reply"]
    advanced_answer = advanced.reply("dungct", "thread-b", "Mình tên gì?")["reply"]

    assert "DũngCT" not in baseline_answer
    assert "DũngCT" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    long_message = "Mình đang viết benchmark memory systems và muốn giữ ngữ cảnh ổn định. " + ("y" * 120)
    for index in range(14):
        message = f"{long_message} lượt {index}"
        baseline.reply("dungct", "thread-long", message)
        advanced.reply("dungct", "thread-long", message)

    assert advanced.compaction_count("thread-long") > 0
    assert advanced.prompt_token_usage("thread-long") < baseline.prompt_token_usage("thread-long")
