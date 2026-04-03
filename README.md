# Madogiwa

# このリポジトリは何か

このリポジトリは、llm-jp/llm-jp-4-8bのバリアントniryuu/madogiwa-8b(仮)を作る試みである。
llm-jp-4-8b-thinkingは、gpt-oss-120bのreasoning traceを用いてSFTされた後、おそらくそのモデルの出力からgpt-oss-120bによりchosen/rejectedを選び、DPOをかけている。
しかし、同日Gemma 4がApache 2ライセンスでリリースされたことにより、gpt-ossを超える可能性のあるreasoning traceを手に入れることができる。
それをとりあえずSFTだけで行おうとするのが、madogiwa-8bである。

# 開発環境

Macで行う。ライブラリは未定である。
