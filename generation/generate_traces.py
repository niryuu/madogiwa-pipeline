"""
extract_prompts.py で生成したJSONLを受け取り、
mlx-vlm + gemma-4-31b-it でthinkingトレースを生成してJSONL形式で保存する。

出力形式 (1行ごと):
{
  "id": "...",
  "subset": "...",
  "split": "...",
  "messages": [
    {"role": "system", "content": ...},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "channel": "analysis"},
    {"role": "assistant", "content": "...", "channel": "final"}
  ]
}
"""

import argparse
import json
import sys

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


def build_prompt_text(conversations):
    """会話リストから最終的なユーザープロンプトのテキストを組み立てる。"""
    # マルチターンの場合は全ユーザー発話を結合
    parts = [turn["content"] for turn in conversations if turn["role"] == "user"]
    return "\n\n".join(parts)


def generate_trace(model, processor, config, user_prompt, max_tokens=4096, temp=1.0):
    """
    gemma-4-31b-it で thinking トレースつき応答を生成する。
    enable_thinking=True で <|think|> トークンを有効化し、
    思考チャネル (<|channel>thought ... <channel|>) を出力させる。
    """
    formatted = apply_chat_template(
        processor, config, user_prompt, enable_thinking=True
    )
    result = generate(
        model,
        processor,
        formatted,
        max_tokens=max_tokens,
        temp=temp,
        verbose=False,
    )
    return result.text


def parse_thinking_response(response_text):
    """
    Gemma 4 の応答を thinking (analysis) 部分と final 部分に分割する。
    Gemma 4 は <|channel>thought ... <channel|> マーカーで思考チャネルを示す。
    """
    analysis_parts = []
    final_parts = []

    # <channel|> で分割し、各パートに <|channel>thought が含まれるか判定
    parts = response_text.split("<channel|>")
    for part in parts:
        if "<|channel>" in part:
            before_channel, channel_rest = part.split("<|channel>", 1)
            # <|channel> の前のテキストは final に属する
            if before_channel.strip():
                final_parts.append(before_channel.strip())
            # <|channel> の後のテキストはチャネル名 + 内容
            # 例: "thought\n内容..." → チャネル名 = "thought", 内容 = "..."
            if channel_rest.startswith("thought"):
                thought_content = channel_rest[len("thought"):].strip()
                if thought_content:
                    analysis_parts.append(thought_content)
            else:
                # thought 以外のチャネルは final に入れる
                if channel_rest.strip():
                    final_parts.append(channel_rest.strip())
        else:
            if part.strip():
                final_parts.append(part.strip())

    analysis = "\n\n".join(analysis_parts)
    final = "\n\n".join(final_parts)
    return analysis, final


def process_record(model, processor, config, record, max_tokens, temp):
    """1レコードを処理してトレースつきメッセージを生成する。"""
    user_prompt = build_prompt_text(record["conversations"])

    response = generate_trace(model, processor, config, user_prompt, max_tokens, temp)
    analysis, final = parse_thinking_response(response)

    # 元のデータセットのメッセージ形式に合わせて構築
    messages = []

    # system
    if record.get("system_content"):
        messages.append({
            "role": "system",
            "content": record["system_content"],
        })

    # user turns
    for turn in record["conversations"]:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": turn["content"]}],
        })

    # assistant: analysis (thinking trace)
    if analysis:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": analysis}],
            "channel": "analysis",
        })

    # assistant: final
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": final}],
        "channel": "final",
    })

    return {
        "id": record["id"],
        "subset": record["subset"],
        "split": record["split"],
        "messages": messages,
    }


def main():
    parser = argparse.ArgumentParser(
        description="mlx-vlm + gemma-4-31b-it でthinkingトレースを生成"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="extract_prompts.py の出力JSONL"
    )
    parser.add_argument(
        "--output", "-o", default="./data/traces.jsonl", help="出力ファイルパス"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-26b-a4b-it-bf16",
        help="使用するMLXモデル (default: mlx-community/gemma-4-31b-it-4bit)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="最大生成トークン数"
    )
    parser.add_argument(
        "--temp", type=float, default=0.7, help="サンプリング温度"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="処理開始インデックス (リジューム用)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="処理する最大レコード数"
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model} ...")
    model, processor = load(args.model)
    config = load_config(args.model)
    print("Model loaded.")

    # 入力を読み込み
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # スライス
    records = records[args.start:]
    if args.limit is not None:
        records = records[:args.limit]

    print(f"Processing {len(records)} records ...")

    mode = "a" if args.start > 0 else "w"
    with open(args.output, mode, encoding="utf-8") as f:
        for i, record in enumerate(records):
            idx = args.start + i
            try:
                result = process_record(
                    model, processor, config, record, args.max_tokens, args.temp
                )
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

                if (i + 1) % 10 == 0:
                    print(f"  [{idx + 1}] {record['id']} done")

            except Exception as e:
                print(f"  [{idx + 1}] ERROR {record['id']}: {e}", file=sys.stderr)
                # エラーでも続行、エラーレコードはスキップ
                continue

    print(f"\nDone. Output saved to {args.output}")


if __name__ == "__main__":
    main()
