"""
SDFT 学習エントリポイント。

  uv run -- python train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import yaml
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load as load_model
from tqdm import tqdm

from data import load_dataset
from sdft_trainer import SDFTTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    # --- 乱数シード ---
    seed = cfg["data"].get("seed", 42)
    random.seed(seed)
    mx.random.seed(seed)

    # --- model + tokenizer ---
    print(f"Loading model: {cfg['model_path']}")
    model, tokenizer = load_model(cfg["model_path"])

    # --- LoRA ---
    lora_cfg = cfg["lora"]
    print(f"Applying LoRA (last {lora_cfg['num_layers']} layers, rank {lora_cfg['rank']})")
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=lora_cfg["num_layers"],
        config={
            "rank": lora_cfg["rank"],
            "scale": lora_cfg["scale"],
            "dropout": lora_cfg["dropout"],
        },
    )

    # 必要なら resume
    resume = cfg["runtime"].get("resume_from")
    if resume:
        print(f"Resuming LoRA adapter from {resume}")
        model.load_weights(resume, strict=False)

    n_trainable = sum(p.size for p in _flat_arrays(model.trainable_parameters()))
    n_total = sum(p.size for p in _flat_arrays(model.parameters()))
    print(f"Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    # --- data ---
    bridge = cfg["sdft"]["bridge_instruction"]
    print(f"Loading dataset: {cfg['data']['train_file']}")
    records = load_dataset(cfg["data"]["train_file"], bridge_instruction=bridge)
    print(f"  loaded {len(records)} records")
    if cfg["data"].get("shuffle", True):
        random.shuffle(records)

    # --- trainer ---
    trainer = SDFTTrainer(model, tokenizer, cfg)

    # --- output ---
    output_dir = Path(cfg["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    log_f = open(log_path, "a", encoding="utf-8")

    # --- training loop ---
    total_steps = cfg["schedule"]["total_steps"]
    log_every = cfg["runtime"]["log_every"]
    save_every = cfg["runtime"]["save_every"]
    batch_size = cfg["data"].get("batch_size", 1)

    pbar = tqdm(total=total_steps, desc="train")
    step_idx = 0
    record_idx = 0
    while step_idx < total_steps:
        # batch を構築
        batch_records = []
        while len(batch_records) < batch_size:
            if record_idx >= len(records):
                random.shuffle(records)
                record_idx = 0
            batch_records.append(records[record_idx])
            record_idx += 1

        t0 = time.time()
        stats_list = trainer.step_batch(batch_records, step_idx)
        elapsed = time.time() - t0

        # batch 内全 record が invalid (overlong 等) なら loss は 0 のまま。skip。
        # SFT-only モード (sdft_weight=0) では completion_len は 0 だが loss は動く。
        if stats_list[0].loss == 0.0 and all(s.completion_len < 2 for s in stats_list):
            continue

        # batch 平均の代表値で log
        rep = stats_list[0]
        if step_idx % log_every == 0:
            log = {
                "step": step_idx,
                "loss": rep.loss,
                "batch_size": len(batch_records),
                "mean_completion_len": sum(s.completion_len for s in stats_list) / len(stats_list),
                "max_student_seq_len": max(s.student_seq_len for s in stats_list),
                "max_teacher_seq_len": max(s.teacher_seq_len for s in stats_list),
                "elapsed_s": elapsed,
                "elapsed_per_record_s": elapsed / len(batch_records),
                "ids": [r.id for r in batch_records],
            }
            log_f.write(json.dumps(log) + "\n")
            log_f.flush()
            pbar.set_postfix({
                "loss": f"{rep.loss:.4f}",
                "y_len": int(log["mean_completion_len"]),
                "s/rec": f"{log['elapsed_per_record_s']:.1f}",
            })

        if save_every > 0 and (step_idx + 1) % save_every == 0:
            ckpt_path = output_dir / f"adapter_step_{step_idx + 1}.safetensors"
            print(f"\n[ckpt] saving to {ckpt_path}")
            mx.save_safetensors(str(ckpt_path), dict(_flat_named(model.trainable_parameters())))

        pbar.update(1)
        step_idx += 1

    pbar.close()
    log_f.close()

    # 最終チェックポイント
    final_path = output_dir / "adapter_final.safetensors"
    print(f"[final] saving to {final_path}")
    mx.save_safetensors(str(final_path), dict(_flat_named(model.trainable_parameters())))


def _flat_arrays(tree):
    """parameter tree から mx.array を再帰的に取り出す。"""
    if isinstance(tree, mx.array):
        yield tree
    elif isinstance(tree, dict):
        for v in tree.values():
            yield from _flat_arrays(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            yield from _flat_arrays(v)


def _flat_named(tree, prefix=""):
    """parameter tree → {dotted_name: array} のジェネレータ。"""
    if isinstance(tree, mx.array):
        yield prefix.rstrip("."), tree
    elif isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flat_named(v, f"{prefix}{k}.")
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            yield from _flat_named(v, f"{prefix}{i}.")


if __name__ == "__main__":
    main()
