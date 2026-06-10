"""
学習済み LoRA adapter を読み込んで生成。

使い方:
  uv run -- python infer.py --adapter ./checkpoints_n1000/adapter_final.safetensors --prompt "1+1は?"
  uv run -- python infer.py --adapter ./checkpoints_n1000/adapter_final.safetensors --from-data 5
  # adapter なしで base model だけ:
  uv run -- python infer.py --prompt "1+1は?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import yaml
from mlx_lm.generate import generate_step
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load as load_model

from data import _msg, START, CHANNEL, MESSAGE, render_system, load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="config.yaml")
    p.add_argument("--adapter", "-a", default=None,
                   help="LoRA adapter safetensors path (なしなら base model)")
    p.add_argument("--prompt", "-p", default=None, help="user prompt")
    p.add_argument("--from-data", type=int, default=None,
                   help="train_file から先頭 N 件を試す")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    return p.parse_args()


def build_prompt(user_text: str, system_content: dict | None = None) -> str:
    """Harmony 形式で student と同じ構造のプロンプトを作る。"""
    sys_block = render_system(system_content)
    return (
        sys_block
        + _msg("user", user_text)
        + f"{START}assistant{CHANNEL}analysis{MESSAGE}"
    )


def generate_text(model, tokenizer, prompt: str, max_tokens: int, temp: float, top_p: float) -> str:
    """Harmony の停止トークンで自動停止しつつ analysis + final まで生成。"""
    from sdft_trainer import _sample_with_top_p
    sampler = lambda logits: _sample_with_top_p(logits, temp=temp, top_p=top_p)

    prompt_ids = mx.array(tokenizer.encode(prompt))

    # 停止トークンセット
    stop_ids = set()
    for tok in ("<|return|>", "<|endoftext|>"):
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) == 1:
            stop_ids.add(ids[0])
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    out_ids = []
    for token, _ in generate_step(prompt_ids, model, sampler=sampler, max_tokens=max_tokens):
        tid = int(token.item()) if hasattr(token, "item") else int(token)
        out_ids.append(tid)
        if tid in stop_ids:
            break

    return tokenizer.decode(out_ids)


def split_channels(text: str) -> dict:
    """生成テキストを Harmony チャネルごとに分解。"""
    parts = {"analysis": "", "final": "", "raw": text}
    cur_channel = "analysis"  # prompt が analysis open で終わってるので最初は analysis
    cur_text = []

    i = 0
    while i < len(text):
        if text[i:i+9] == "<|start|>":
            # 現在の channel に積みあげたものを保存
            parts[cur_channel] = parts[cur_channel] + "".join(cur_text)
            cur_text = []
            i += 9
            # ロール部分をスキップ (assistant)
            end_role = text.find("<|", i)
            if end_role == -1:
                break
            # channel 指定があれば取得
            tag = text[i:end_role]
            i = end_role
            if text[i:i+11] == "<|channel|>":
                i += 11
                end_ch = text.find("<|message|>", i)
                if end_ch == -1:
                    break
                cur_channel = text[i:end_ch].strip()
                i = end_ch + len("<|message|>")
            elif text[i:i+11] == "<|message|>":
                i += 11
                cur_channel = "(other)"
        elif text[i:i+7] == "<|end|>":
            parts[cur_channel] = parts[cur_channel] + "".join(cur_text)
            cur_text = []
            i += 7
        else:
            cur_text.append(text[i])
            i += 1

    parts[cur_channel] = parts[cur_channel] + "".join(cur_text)
    return parts


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    print(f"Loading model: {cfg['model_path']}")
    model, tokenizer = load_model(cfg["model_path"])

    if args.adapter:
        print(f"Applying LoRA layout (matching training: last {cfg['lora']['num_layers']} layers, rank {cfg['lora']['rank']})")
        model.freeze()
        linear_to_lora_layers(
            model,
            num_layers=cfg["lora"]["num_layers"],
            config={
                "rank": cfg["lora"]["rank"],
                "scale": cfg["lora"]["scale"],
                "dropout": cfg["lora"]["dropout"],
            },
        )
        print(f"Loading adapter: {args.adapter}")
        model.load_weights(args.adapter, strict=False)
    else:
        print("(no adapter — base model)")

    # 推論プロンプト準備
    prompts = []
    if args.from_data is not None:
        bridge = cfg["sdft"]["bridge_instruction"]
        records = load_dataset(cfg["data"]["train_file"], bridge_instruction=bridge)
        for r in records[:args.from_data]:
            user_text = r.user_prompts[0]
            sys_content = None  # 元の system は使わずデフォルトで良い
            prompts.append({
                "id": r.id,
                "user": user_text,
                "demo_analysis": r.demo_analysis[:300] + ("..." if len(r.demo_analysis) > 300 else ""),
                "demo_final": r.demo_final,
                "harmony": build_prompt(user_text, sys_content),
            })
    elif args.prompt:
        prompts.append({
            "id": "(cli)",
            "user": args.prompt,
            "demo_analysis": None,
            "demo_final": None,
            "harmony": build_prompt(args.prompt, None),
        })
    else:
        # デフォルトでテストプロンプト数件
        defaults = [
            "次の英文を日本語に訳してください: 'The cat sat on the mat.'",
            "10進数の42を2進数に変換してください。",
            "「猫」と「犬」の違いを3点で説明してください。",
        ]
        for p in defaults:
            prompts.append({
                "id": "(default)",
                "user": p,
                "demo_analysis": None,
                "demo_final": None,
                "harmony": build_prompt(p, None),
            })

    for i, p in enumerate(prompts):
        print()
        print("=" * 70)
        print(f"[{i+1}/{len(prompts)}] id={p['id']}")
        print(f"USER:\n{p['user']}")
        if p.get("demo_final"):
            print(f"\nGEMMA DEMO FINAL:\n{p['demo_final']}")
        print()
        print("--- generating ---")
        text = generate_text(model, tokenizer, p["harmony"],
                             max_tokens=args.max_tokens, temp=args.temp, top_p=args.top_p)
        parsed = split_channels(text)
        print(f"\nANALYSIS:\n{parsed['analysis'].strip()}")
        print(f"\nFINAL:\n{parsed['final'].strip()}")


if __name__ == "__main__":
    main()
