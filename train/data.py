"""
data/traces_*_clean.jsonl を SDFT 用に整形する。

Harmony フォーマット (analysis + final サブセット) で:
  - student_prompt: prompt のみ。analysis チャネルを開いた状態で停止 (sampling 用)
  - teacher_prefix: prompt + demo_analysis + demo_final + bridge instruction +
                    analysis チャネルを開いた状態で停止 (KL 計算用 prefix)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# --- Harmony tokens ---
START = "<|start|>"
END = "<|end|>"
MESSAGE = "<|message|>"
CHANNEL = "<|channel|>"


def _msg(role: str, content: str, channel: str | None = None) -> str:
    if channel is not None:
        return f"{START}{role}{CHANNEL}{channel}{MESSAGE}{content}{END}"
    return f"{START}{role}{MESSAGE}{content}{END}"


def render_system(system_content: dict | None) -> str:
    """元データの system_content (dict) を Harmony system message に変換。"""
    parts: list[str] = []
    if system_content:
        if "model_identity" in system_content:
            parts.append(system_content["model_identity"])
        if "knowledge_cutoff" in system_content:
            parts.append(f"Knowledge cutoff: {system_content['knowledge_cutoff']}")
        if "conversation_start_date" in system_content:
            parts.append(f"Current date: {system_content['conversation_start_date']}")
        if "reasoning_effort" in system_content:
            parts.append(f"Reasoning: {system_content['reasoning_effort'].lower()}")
    parts.append("# Valid channels: analysis, final")
    return _msg("system", "\n".join(parts))


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


@dataclass
class SDFTRecord:
    id: str
    student_prompt: str   # x: prompt + assistant<analysis>open
    teacher_prefix: str   # (x + c + bridge): demo を見た後の assistant<analysis>open

    # 整形前の生データ (デバッグ用)
    user_prompts: list[str]
    demo_analysis: str
    demo_final: str


def parse_record(raw: dict, bridge_instruction: str) -> SDFTRecord | None:
    system_content: dict | None = None
    user_prompts: list[str] = []
    demo_analysis = ""
    demo_final = ""

    for msg in raw.get("messages", []):
        role = msg.get("role", "")
        if role == "system":
            content = msg.get("content")
            if isinstance(content, dict):
                system_content = content
            elif isinstance(content, list) and content and isinstance(content[0], dict):
                system_content = content[0]
        elif role == "user":
            user_prompts.append(_extract_text(msg.get("content", "")))
        elif role == "assistant":
            channel = msg.get("channel", "")
            text = _extract_text(msg.get("content", ""))
            if channel == "analysis":
                demo_analysis = text
            elif channel == "final":
                demo_final = text

    if not user_prompts or not demo_analysis.strip() or not demo_final.strip():
        return None

    sys_block = render_system(system_content)
    user_blocks = "".join(_msg("user", p) for p in user_prompts)

    # SDFT 論文の prompt template を Harmony 化したもの。
    # student は demo を見ずに analysis から書く。
    # teacher は demo (analysis + final) を見たうえで、bridge 指示を経て改めて analysis から書く。
    student_prompt = (
        sys_block
        + user_blocks
        + f"{START}assistant{CHANNEL}analysis{MESSAGE}"
    )
    teacher_prefix = (
        sys_block
        + user_blocks
        + _msg("assistant", demo_analysis, "analysis")
        + _msg("assistant", demo_final, "final")
        + _msg("user", bridge_instruction)
        + f"{START}assistant{CHANNEL}analysis{MESSAGE}"
    )

    return SDFTRecord(
        id=raw.get("id", ""),
        student_prompt=student_prompt,
        teacher_prefix=teacher_prefix,
        user_prompts=user_prompts,
        demo_analysis=demo_analysis,
        demo_final=demo_final,
    )


def load_dataset(
    path: str | Path,
    bridge_instruction: str,
) -> list[SDFTRecord]:
    records: list[SDFTRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rec = parse_record(raw, bridge_instruction)
            if rec is not None:
                records.append(rec)
    return records
