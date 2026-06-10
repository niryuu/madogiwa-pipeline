"""
SDFT trainer for MLX.

Algorithm (paper 式 1, 2):
  1. Sample on-policy completion y ~ π_θ(·|x) from student
  2. Forward student on (x ++ y), forward teacher on (x ++ c ++ bridge ++ y)
  3. Reverse KL: D_KL( π_θ(·|x) || π_teacher(·|x, c) ) over completion tokens
  4. Backprop; AdamW step on student LoRA params
  5. EMA: teacher_lora ← (1-α) teacher_lora + α student_lora
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from mlx_lm.generate import generate_step

from data import SDFTRecord, START, CHANNEL, MESSAGE


@dataclass
class StepStats:
    loss: float
    kl_loss: float
    sft_loss: float
    completion_len: int
    student_seq_len: int
    teacher_seq_len: int


def _flatten_named(tree, prefix=""):
    """Flatten parameter tree to {dotted_name: array} for explicit keying."""
    flat = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            flat.update(_flatten_named(v, f"{prefix}{k}.") if prefix else _flatten_named(v, f"{k}."))
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            flat.update(_flatten_named(v, f"{prefix}{i}."))
    elif isinstance(tree, mx.array):
        flat[prefix.rstrip(".")] = tree
    return flat


def reverse_kl(student_logits: mx.array, teacher_logits: mx.array, temperature: float = 1.0) -> mx.array:
    """
    Reverse KL: sum_v p_student(v) * (log p_student(v) - log p_teacher(v))

    Args:
        student_logits: [T, V]
        teacher_logits: [T, V]
    Returns:
        scalar: token-平均された KL
    """
    s = student_logits / temperature
    t = teacher_logits / temperature
    s_logp = nn.log_softmax(s, axis=-1)
    t_logp = nn.log_softmax(t, axis=-1)
    s_p = mx.exp(s_logp)
    kl_per_token = (s_p * (s_logp - t_logp)).sum(axis=-1)  # [T]
    return kl_per_token.mean()


class SDFTTrainer:
    def __init__(self, model, tokenizer, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # 学生 LoRA params のスナップショットを取って teacher の初期値とする
        # (= train 開始時、teacher == student)
        # MLX arrays は immutable なので参照共有で十分。dict 構造だけ独立させる。
        student_lora = self.model.trainable_parameters()
        self.teacher_lora = tree_map(lambda x: x, student_lora)

        # optimizer
        opt_cfg = config["optimizer"]
        self.optimizer = optim.AdamW(
            learning_rate=self._make_lr_schedule(),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        # 停止トークン: Harmony の <|end|> + EOS
        self.stop_ids: set[int] = set()
        for tok in ("<|end|>", "<|return|>", "<|endoftext|>"):
            try:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                if len(ids) == 1:
                    self.stop_ids.add(ids[0])
            except Exception:
                pass
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            self.stop_ids.add(tokenizer.eos_token_id)

    def _make_lr_schedule(self):
        sched = self.config["schedule"]
        opt = self.config["optimizer"]
        warmup = sched.get("warmup_steps", 0)
        total = sched.get("total_steps", 1)
        peak_lr = opt["learning_rate"]

        if warmup > 0:
            warmup_sched = optim.linear_schedule(0.0, peak_lr, warmup)
            decay_sched = optim.cosine_decay(peak_lr, max(1, total - warmup))
            return optim.join_schedules([warmup_sched, decay_sched], [warmup])
        return optim.cosine_decay(peak_lr, total)

    # --- LoRA swap ---
    def _snapshot_student_lora(self) -> dict:
        return tree_map(lambda x: x, self.model.trainable_parameters())

    def _load_lora(self, lora_tree: dict) -> None:
        self.model.update(lora_tree)

    # --- on-policy sampling ---
    def sample_completion(self, prompt_text: str) -> mx.array:
        prompt_ids = mx.array(self.tokenizer.encode(prompt_text))
        s_cfg = self.config["sampling"]
        max_tokens = s_cfg["max_completion_tokens"]
        temp = s_cfg["temperature"]
        top_p = s_cfg["top_p"]

        out_ids: list[int] = []
        # generate_step は (token, logprobs) を yield する。
        # generate_step 自体に max_tokens=256 のデフォルトがあるので明示指定が必須。
        sampler = lambda logits: _sample_with_top_p(logits, temp=temp, top_p=top_p)
        for token, _ in generate_step(
            prompt_ids, self.model, sampler=sampler, max_tokens=max_tokens
        ):
            tid = int(token.item()) if hasattr(token, "item") else int(token)
            out_ids.append(tid)
            if tid in self.stop_ids:
                break
        return mx.array(out_ids, dtype=mx.int32)

    def _pad_id(self) -> int:
        """forward 時の右 padding 用トークン ID。"""
        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            return int(self.tokenizer.pad_token_id)
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            return int(self.tokenizer.eos_token_id)
        return 0

    def _right_pad_batch(self, sequences: list[mx.array], pad_id: int) -> mx.array:
        """[T_i] の list を [B, max(T_i)] に右 padding。"""
        max_len = max(int(s.shape[0]) for s in sequences)
        rows = []
        for s in sequences:
            sl = int(s.shape[0])
            if sl < max_len:
                pad = mx.full((max_len - sl,), pad_id, dtype=mx.int32)
                rows.append(mx.concatenate([s, pad]))
            else:
                rows.append(s)
        return mx.stack(rows, axis=0)

    # --- step (single record) ---
    def step(self, record: SDFTRecord, step_idx: int) -> StepStats:
        stats = self.step_batch([record], step_idx)
        return stats[0]

    def _build_sft_target(self, record: SDFTRecord) -> str:
        """SFT 対象テキスト = demo_analysis<|end|> + assistant<final>open + demo_final<|end|>。
        student_prompt の末尾が assistant<analysis><|message|> なので、これに直接続けられる。
        """
        return (
            record.demo_analysis
            + "<|end|>"
            + f"{START}assistant{CHANNEL}final{MESSAGE}"
            + record.demo_final
            + "<|end|>"
        )

    # --- step (batched) ---
    def step_batch(self, records: list[SDFTRecord], step_idx: int) -> list[StepStats]:
        """
        B records をまとめて 1 step 学習する。
        loss = sdft_weight * KL + sft_weight * CE
        sdft_weight=0 → 純 SFT (生成・teacher forward 省略、高速)
        sft_weight=0  → 純 SDFT (旧挙動)
        """
        cfg = self.config
        B = len(records)
        max_len = cfg["data"]["max_seq_len"]
        pad_id = self._pad_id()

        sft_weight = float(cfg["sdft"].get("sft_weight", 0.0))
        sdft_weight = 1.0 - sft_weight
        kl_temp = cfg["sdft"].get("temperature", 1.0)

        # 1. tokenize
        student_prompts_ids = [
            mx.array(self.tokenizer.encode(r.student_prompt), dtype=mx.int32)
            for r in records
        ]

        # --- SFT 入力構築 (sft_weight > 0 のとき) ---
        sft_inputs_list: list[mx.array] = []
        sft_target_ranges: list[tuple[int, int]] = []  # (start, len)
        sft_valid: list[bool] = []
        for sp_ids, r in zip(student_prompts_ids, records):
            target_str = self._build_sft_target(r)
            target_ids = mx.array(self.tokenizer.encode(target_str), dtype=mx.int32)
            start = int(sp_ids.shape[0])
            tlen = int(target_ids.shape[0])
            valid = tlen >= 2 and (start + tlen) <= max_len
            if valid:
                sft_inputs_list.append(mx.concatenate([sp_ids, target_ids]))
                sft_target_ranges.append((start, tlen))
            else:
                # overlong / invalid record: ダミー (sp_ids のみ) を入れて batch shape 抑制
                sft_inputs_list.append(sp_ids)
                sft_target_ranges.append((start, 0))
            sft_valid.append(valid)

        # --- SDFT パスの準備 (sdft_weight > 0 のとき) ---
        teacher_prefixes_ids = []
        completions = []
        per_record_lens = []
        student_inputs_list = []
        teacher_inputs_list = []
        teacher_y_logits_list: list = []
        student_batch = None
        teacher_batch = None

        if sdft_weight > 0:
            teacher_prefixes_ids = [
                mx.array(self.tokenizer.encode(r.teacher_prefix), dtype=mx.int32)
                for r in records
            ]

            # on-policy sampling
            for r in records:
                completions.append(self.sample_completion(r.student_prompt))

            for i, r in enumerate(records):
                sp_len = int(student_prompts_ids[i].shape[0])
                tp_len = int(teacher_prefixes_ids[i].shape[0])
                c_len = int(completions[i].shape[0])
                s_full = sp_len + c_len
                t_full = tp_len + c_len
                valid = c_len >= 2 and s_full <= max_len and t_full <= max_len
                per_record_lens.append({
                    "s_y_start": sp_len - 1,
                    "t_y_start": tp_len - 1,
                    "y_len": c_len,
                    "s_full": s_full,
                    "t_full": t_full,
                    "valid": valid,
                })
                if valid:
                    student_inputs_list.append(
                        mx.concatenate([student_prompts_ids[i], completions[i]])
                    )
                    teacher_inputs_list.append(
                        mx.concatenate([teacher_prefixes_ids[i], completions[i]])
                    )
                else:
                    student_inputs_list.append(student_prompts_ids[i])
                    teacher_inputs_list.append(teacher_prefixes_ids[i])

            student_batch = self._right_pad_batch(student_inputs_list, pad_id)
            teacher_batch = self._right_pad_batch(teacher_inputs_list, pad_id)

            # teacher forward (no grad, LoRA swap)
            student_lora_backup = self._snapshot_student_lora()
            self._load_lora(self.teacher_lora)
            teacher_logits_full = self.model(teacher_batch)
            for i, p in enumerate(per_record_lens):
                if p["valid"]:
                    teacher_y_logits_list.append(
                        mx.stop_gradient(
                            teacher_logits_full[i, p["t_y_start"]:p["t_y_start"] + p["y_len"]]
                        )
                    )
                else:
                    teacher_y_logits_list.append(None)
            self._load_lora(student_lora_backup)
            del student_lora_backup, teacher_logits_full
            for t in teacher_y_logits_list:
                if t is not None:
                    mx.eval(t)
        else:
            # 純 SFT: dummy stats
            for i in range(B):
                sp_len = int(student_prompts_ids[i].shape[0])
                per_record_lens.append({
                    "s_y_start": -1, "t_y_start": -1, "y_len": 0,
                    "s_full": sp_len, "t_full": 0, "valid": False,
                })

        # SFT batch (sft_weight > 0 のとき)
        sft_batch = None
        if sft_weight > 0:
            sft_batch = self._right_pad_batch(sft_inputs_list, pad_id)

        any_valid_sdft = any(p["valid"] for p in per_record_lens)
        any_valid_sft = any(sft_valid)

        if not any_valid_sdft and not any_valid_sft:
            return [
                StepStats(loss=0.0, kl_loss=0.0, sft_loss=0.0,
                          completion_len=p["y_len"],
                          student_seq_len=p["s_full"], teacher_seq_len=p["t_full"])
                for p in per_record_lens
            ]

        # --- loss + backward ---
        def loss_fn(model):
            kl_part = mx.array(0.0)
            ce_part = mx.array(0.0)

            if sdft_weight > 0 and any_valid_sdft:
                student_logits_full = model(student_batch)
                n_valid = 0
                for i, p in enumerate(per_record_lens):
                    if not p["valid"]:
                        continue
                    s_y_logits = student_logits_full[
                        i, p["s_y_start"]:p["s_y_start"] + p["y_len"]
                    ]
                    kl_part = kl_part + reverse_kl(
                        s_y_logits, teacher_y_logits_list[i], temperature=kl_temp
                    )
                    n_valid += 1
                kl_part = kl_part / max(n_valid, 1)

            if sft_weight > 0 and any_valid_sft:
                sft_logits_full = model(sft_batch)
                n_valid = 0
                for i in range(B):
                    if not sft_valid[i]:
                        continue
                    start, tlen = sft_target_ranges[i]
                    # logit[t] が token[t+1] を予測
                    logits_for_targets = sft_logits_full[i, start - 1:start + tlen - 1]
                    target_ids = sft_inputs_list[i][start:start + tlen]
                    ce_part = ce_part + nn.losses.cross_entropy(
                        logits_for_targets, target_ids, reduction="mean"
                    )
                    n_valid += 1
                ce_part = ce_part / max(n_valid, 1)

            total = sdft_weight * kl_part + sft_weight * ce_part
            return total, kl_part, ce_part

        # value_and_grad は scalar を返す関数を期待するので工夫
        def total_loss_fn(model):
            total, _, _ = loss_fn(model)
            return total

        # 別途 KL/CE を出すために 1 度だけ評価する
        loss, grads = nn.value_and_grad(self.model, total_loss_fn)(self.model)
        # KL, CE は backprop 後にもう一度 forward 評価で取得 (重い場合は省略可能)
        # → コスト節約のため、loss_fn の戻り値を再利用したい。
        # ここでは簡略化として、loss_fn を再呼び出しせず、loss_val のみ返す。

        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)

        # EMA update teacher (SDFT 側のみ意味があるが、無条件に更新しても害なし)
        alpha = cfg["sdft"]["ema_alpha"]
        current_student = self.model.trainable_parameters()
        self.teacher_lora = tree_map(
            lambda t, s: (1.0 - alpha) * t + alpha * s,
            self.teacher_lora,
            current_student,
        )
        mx.eval(self.teacher_lora)

        loss_val = float(loss.item())
        # 個別の KL/CE は概算 (loss_val から逆算は難しいので 0 で埋める)
        return [
            StepStats(loss=loss_val, kl_loss=0.0, sft_loss=0.0,
                      completion_len=p["y_len"],
                      student_seq_len=p["s_full"], teacher_seq_len=p["t_full"])
            for p in per_record_lens
        ]


def _sample_with_top_p(logits: mx.array, temp: float, top_p: float) -> mx.array:
    """top-p + temperature サンプリング。logits: [V] or [B, V]."""
    if temp <= 0.0:
        return mx.argmax(logits, axis=-1)
    scaled = logits / temp
    probs = mx.softmax(scaled, axis=-1)

    if top_p < 1.0:
        sorted_idx = mx.argsort(-probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_idx, axis=-1)
        cum = mx.cumsum(sorted_probs, axis=-1)
        # 累積が top_p を超える位置以降をマスク
        mask = cum - sorted_probs > top_p
        sorted_probs = mx.where(mask, mx.zeros_like(sorted_probs), sorted_probs)
        # normalize
        sorted_probs = sorted_probs / sorted_probs.sum(axis=-1, keepdims=True)
        sampled_sorted_idx = mx.random.categorical(mx.log(sorted_probs + 1e-12))
        return mx.take_along_axis(sorted_idx, sampled_sorted_idx[..., None], axis=-1).squeeze(-1)
    return mx.random.categorical(mx.log(probs + 1e-12))
