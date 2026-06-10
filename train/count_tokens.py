"""
Harmony 整形後のデータを llm-jp-4 トークナイザで実際にトークン化し、
学習データのトークン数分布を集計する。

学習で実際に loss に効くのは on-policy completion トークンだが、
それは生成依存なのでここでは demo (analysis + final) トークン数を
「データが提供する supervision 信号の総量」として集計する。
"""

import argparse
from pathlib import Path
import statistics

import yaml
from mlx_lm.utils import load as load_model

from data import load_dataset, render_system, _msg, START, CHANNEL, MESSAGE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="config.yaml")
    p.add_argument("--limit", type=int, default=None, help="先頭N件のみ集計")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    print(f"Loading tokenizer from {cfg['model_path']} ...")
    _model, tokenizer = load_model(cfg["model_path"])
    del _model  # tokenizer だけあれば十分

    bridge = cfg["sdft"]["bridge_instruction"]
    print(f"Loading dataset: {cfg['data']['train_file']}")
    records = load_dataset(cfg["data"]["train_file"], bridge_instruction=bridge)
    if args.limit:
        records = records[:args.limit]
    print(f"  {len(records)} records")
    print()

    # トークン数を集計
    student_prompt_tokens = []  # x = system + user (analysis open まで)
    teacher_prefix_tokens = []  # x + c + bridge (analysis open まで)
    demo_analysis_tokens = []   # demo の analysis 部分のみ
    demo_final_tokens = []      # demo の final 部分のみ
    demo_total_tokens = []      # analysis + final + 周りの Harmony マーカー

    for i, rec in enumerate(records):
        student_prompt_tokens.append(len(tokenizer.encode(rec.student_prompt)))
        teacher_prefix_tokens.append(len(tokenizer.encode(rec.teacher_prefix)))

        # demo を Harmony 整形してトークン化 (assistant analysis + assistant final)
        demo_analysis_str = (
            f"{START}assistant{CHANNEL}analysis{MESSAGE}{rec.demo_analysis}<|end|>"
        )
        demo_final_str = (
            f"{START}assistant{CHANNEL}final{MESSAGE}{rec.demo_final}<|end|>"
        )
        a_tok = len(tokenizer.encode(demo_analysis_str))
        f_tok = len(tokenizer.encode(demo_final_str))
        demo_analysis_tokens.append(a_tok)
        demo_final_tokens.append(f_tok)
        demo_total_tokens.append(a_tok + f_tok)

        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1} / {len(records)}")

    print()
    print("=" * 60)
    print(f"Records: {len(records)}")
    print()

    def report(name, data):
        data_sorted = sorted(data)
        n = len(data_sorted)
        total = sum(data_sorted)
        print(f"[{name}]")
        print(f"  total : {total:>15,} tokens")
        print(f"  mean  : {total/n:>15,.1f}")
        print(f"  min   : {data_sorted[0]:>15,}")
        print(f"  p25   : {data_sorted[n//4]:>15,}")
        print(f"  median: {data_sorted[n//2]:>15,}")
        print(f"  p75   : {data_sorted[3*n//4]:>15,}")
        print(f"  p95   : {data_sorted[int(n*0.95)]:>15,}")
        print(f"  max   : {data_sorted[-1]:>15,}")
        print()

    report("student_prompt (system + user + analysis open)", student_prompt_tokens)
    report("teacher_prefix (above + demo + bridge + analysis open)", teacher_prefix_tokens)
    report("demo_analysis (assistant<analysis>...)", demo_analysis_tokens)
    report("demo_final    (assistant<final>...)", demo_final_tokens)
    report("demo_total    (analysis + final)", demo_total_tokens)

    # 学習で実際に loss に効くトークン数の見積もり
    # (注: 実際は on-policy 生成なので demo 長と一致しない。demo を上限の目安として表示)
    print("---")
    print("学習1 epoch (= 全データ走査) でのトークン量目安:")
    print(f"  on-policy 生成上限 256 tok × {len(records)} 件 ≈ {256 * len(records):,} tok (loss計算範囲)")
    print(f"  demo 信号総量 (analysis + final): {sum(demo_total_tokens):,} tok")
    print()
    print(f"今回の 500 step 設定での生成総量目安:")
    cfg_max = cfg["sampling"]["max_completion_tokens"]
    cfg_steps = cfg["schedule"]["total_steps"]
    print(f"  {cfg_max} tok/step × {cfg_steps} step ≈ {cfg_max * cfg_steps:,} tok")


if __name__ == "__main__":
    main()
