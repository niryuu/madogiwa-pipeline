"""
traces JSONL から analysis または final が欠けているレコードを除外する。
"""

import argparse
import json


def _get_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return ""


def has_both_channels(record: dict) -> bool:
    """analysis と final が両方存在し、かつテキストが空でないことを確認する。"""
    found = {}
    for msg in record.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        channel = msg.get("channel", "")
        if channel in ("analysis", "final"):
            found[channel] = _get_text(msg.get("content", "")).strip()
    return bool(found.get("analysis")) and bool(found.get("final"))


def main():
    parser = argparse.ArgumentParser(
        description="analysis/final が欠けたレコードを除外"
    )
    parser.add_argument(
        "--input", "-i", default="./data/traces_10000.jsonl", help="入力ファイル"
    )
    parser.add_argument(
        "--output", "-o", default="./data/traces_10000_clean.jsonl", help="出力ファイル"
    )
    args = parser.parse_args()

    kept = 0
    dropped = 0
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if has_both_channels(record):
                fout.write(line + "\n")
                kept += 1
            else:
                dropped += 1

    print(f"Done. kept={kept} dropped={dropped} -> {args.output}")


if __name__ == "__main__":
    main()
