# generation

llm-jp/llm-jp-4-thinking-sft-data のプロンプトを使い、gemma-4-31b-it で thinking トレースを再生成するスクリプト群。

## セットアップ

```bash
pip install datasets mlx-vlm
```

## 使い方

### 1. プロンプト抽出

`extract_prompts.py` で元データセットからユーザープロンプトを JSONL に抽出する。

```bash
# 全サブセット (reasoning_high split)
python extract_prompts.py -o prompts.jsonl

# サブセット・split を指定
python extract_prompts.py --subsets daring_anteater flan --splits reasoning_high reasoning_medium -o prompts.jsonl
```

出力例:

```json
{"id": "daring_anteater_en-44451", "subset": "daring_anteater", "split": "reasoning_high", "system_content": {...}, "conversations": [{"role": "user", "content": "..."}]}
```

### 2. トレース生成

`generate_traces.py` で mlx-vlm + gemma-4-31b-it を使い thinking トレースを生成する。

```bash
# 基本
python generate_traces.py -i prompts.jsonl -o traces.jsonl

# モデル・パラメータ指定
python generate_traces.py -i prompts.jsonl -o traces.jsonl \
  --model mlx-community/gemma-4-31b-it-8bit \
  --max-tokens 8192 --temp 0.6

# 途中から再開
python generate_traces.py -i prompts.jsonl -o traces.jsonl --start 1000 --limit 500
```

出力は元データセットの messages 形式に合わせ、Gemma 4 の `<think>` タグを `channel: "analysis"` / `channel: "final"` に分離して保存する。

## 利用可能なモデル

| モデル ID | サイズ |
|-----------|--------|
| `mlx-community/gemma-4-31b-it-4bit` (デフォルト) | ~5 GB |
| `mlx-community/gemma-4-31b-it-8bit` | ~9 GB |
| `mlx-community/gemma-4-31b-it-bf16` | ~31 GB |

## 元データセットのサブセット一覧

`daring_anteater`, `flan`, `jaster_v1.4.1`, `llmjp_extraction_wiki_ja_v0.1`, `llmjp_extraction_wiki_ja_v0.2`, `llmjp_extraction_wiki_ja_v0.3`, `llmjp_magpie_sft_v1.0`, `logical_math_coding_wizard8x22b`, `multiturn_calm3`, `nemotron_post_v2_stem`, `nemotron_post_v3_chat`, `nemotron_post_v3_if`, `nemotron_post_v3_math`, `random_to_fixed_multiturn_calm3`, `self_system_question`, `synthetic_jp_en_coding`, `system_prompt_question`
