# train — SDFT for llm-jp-4-base on MLX

`data/traces_*_clean.jsonl` (gemma などの expert traces) を教師信号にして、
`llm-jp/llm-jp-4-8b-base` を SDFT (Self-Distillation Fine-Tuning) で蒸留する。

## アルゴリズム

[Shenfeld et al., 2026 — Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) の SDFT。

- student π_θ: 学習対象 (LoRA wrapped llm-jp-4-base)
- teacher π: student の **EMA コピー** + (prompt + demo) で条件付け
- 損失: 学生サンプル completion 上で reverse KL `D_KL(π_θ(·|x) ‖ π(·|x, c))`
- EMA: 各ステップで `teacher ← (1-α) * teacher + α * student` (α = 0.01)

## データフォーマット (Harmony, analysis + final サブセット)

Student input:
```
<|start|>system<|message|>...<|end|>
<|start|>user<|message|>{prompt}<|end|>
<|start|>assistant<|channel|>analysis<|message|>{student samples here}
```

Teacher input:
```
<|start|>system<|message|>...<|end|>
<|start|>user<|message|>{prompt}<|end|>
<|start|>assistant<|channel|>analysis<|message|>{demo_analysis}<|end|>
<|start|>assistant<|channel|>final<|message|>{demo_final}<|end|>
<|start|>user<|message|>Now answer with a response of your own, including the thinking process.<|end|>
<|start|>assistant<|channel|>analysis<|message|>{student samples here}
```

両 forward の `{student samples here}` 位置の logits に対して reverse KL。

## セットアップ・実行

```bash
cd train
uv sync

# 学習
uv run -- python train.py --config config.yaml
```

`config.yaml` を編集して `model_path`, `lora.num_layers`, `schedule.total_steps`,
`sampling.max_completion_tokens` などを調整する。

### LoRA パラメータ

LoRA を最終 N 層に適用する。M2 Ultra 128GB なら 8B モデル全 32 層に rank 16
LoRA でも余裕があるはず。メモリが厳しければ `num_layers` を減らす。

### 出力

- `checkpoints/train_log.jsonl` — 各ステップの損失・系列長・elapsed time
- `checkpoints/adapter_step_<N>.safetensors` — 中間 LoRA adapter
- `checkpoints/adapter_final.safetensors` — 最終 LoRA adapter

## 既知の制約・TODO

- M2 Ultra 単機での MLX 実装。multi-device には未対応。
- バッチサイズは 1 固定 (grad_accum で調整)。可変長 + Harmony 区切りなのでバッチ化は要設計。
- on-policy サンプリングが学習速度の bottleneck になる見込み。
- 評価ループ未実装。別途 `mlx_lm.generate` で adapter 込みでサンプリングして見る等。
