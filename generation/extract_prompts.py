"""
llm-jp/llm-jp-4-thinking-sft-data からプロンプトを抽出し、JSONL形式で保存する。

出力形式 (1行ごと):
{
  "id": "...",
  "subset": "daring_anteater",
  "split": "reasoning_high",
  "system_content": { ... },
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "user", "content": "..."},  // マルチターンの場合
    ...
  ]
}
"""

import argparse
import json

from datasets import load_dataset


# 全サブセット一覧
SUBSETS = [
    "daring_anteater",
    "flan",
    "jaster_v1.4.1",
    "llmjp_extraction_wiki_ja_v0.1",
    "llmjp_extraction_wiki_ja_v0.2",
    "llmjp_extraction_wiki_ja_v0.3",
    "llmjp_magpie_sft_v1.0",
    "logical_math_coding_wizard8x22b",
    "multiturn_calm3",
    "nemotron_post_v2_stem",
    "nemotron_post_v3_chat",
    "nemotron_post_v3_if",
    "nemotron_post_v3_math",
    "random_to_fixed_multiturn_calm3",
    "self_system_question",
    "synthetic_jp_en_coding",
    "system_prompt_question",
]


def extract_user_turns(messages):
    """messagesからuserのテキストを抽出する。"""
    conversations = []
    for msg in messages:
        if msg["role"] != "user":
            continue
        # contentはリスト形式
        texts = []
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block["text"])
            elif isinstance(block, str):
                texts.append(block)
        if texts:
            conversations.append({"role": "user", "content": "\n".join(texts)})
    return conversations


def extract_system_content(messages):
    """messagesからsystemのcontentを抽出する。"""
    for msg in messages:
        if msg["role"] == "system":
            content = msg.get("content", [])
            if content and isinstance(content, list):
                return content[0] if isinstance(content[0], dict) else {"text": content[0]}
    return None


def process_subset(subset, split, output_path, append=True):
    """1つのsubset/splitを処理してJSONLに追記する。"""
    print(f"Loading {subset}/{split} ...")
    try:
        ds = load_dataset(
            "llm-jp/llm-jp-4-thinking-sft-data",
            name=subset,
            split=split,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Skipping {subset}/{split}: {e}")
        return 0

    mode = "a" if append else "w"
    count = 0
    with open(output_path, mode, encoding="utf-8") as f:
        for row in ds:
            messages = row["messages"]
            if isinstance(messages, str):
                messages = json.loads(messages)

            conversations = extract_user_turns(messages)
            if not conversations:
                continue

            record = {
                "id": row.get("ID", row.get("id", "")),
                "subset": subset,
                "split": split,
                "system_content": extract_system_content(messages),
                "conversations": conversations,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Extracted {count} prompts from {subset}/{split}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="llm-jp-4-thinking-sft-data からプロンプトを抽出"
    )
    parser.add_argument(
        "--output", "-o", default="prompts.jsonl", help="出力ファイルパス"
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="処理するサブセット (未指定で全サブセット)",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["reasoning_high"],
        help="処理するsplit (default: reasoning_high)",
    )
    args = parser.parse_args()

    subsets = args.subsets if args.subsets else SUBSETS

    # 最初は上書き、以降は追記
    first = True
    total = 0
    for subset in subsets:
        for split in args.splits:
            count = process_subset(subset, split, args.output, append=not first)
            total += count
            first = False

    print(f"\nDone. Total {total} prompts saved to {args.output}")


if __name__ == "__main__":
    main()
