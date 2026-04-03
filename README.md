# Madogiwa

# このリポジトリは何か

このリポジトリは、[llm-jp/llm-jp-4-8b](https://huggingface.co/llm-jp/llm-jp-4-8b-base)のバリアントniryuu/madogiwa-8b(仮)を作る試みである。

[llm-jp-4-8b-thinking](https://huggingface.co/llm-jp/llm-jp-4-8b-thinking)は、[gpt-oss-120bのreasoning traceを用いてSFTされた](https://huggingface.co/datasets/llm-jp/llm-jp-4-thinking-sft-data)後、おそらく[そのモデルの出力からgpt-oss-120bによりchosen/rejectedを選び、DPOをかけている。](https://huggingface.co/datasets/llm-jp/llm-jp-4-8b-thinking-dpo-data)
しかし、同日[Gemma 4](https://huggingface.co/google/gemma-4-31B-it)がApache 2ライセンスでリリースされたことにより、gpt-ossを超える可能性のあるreasoning traceを手に入れることができる。
それをとりあえずSFTだけで行おうとするのが、madogiwa-8bである。

# 開発環境

Macで行う。ライブラリは未定である。
