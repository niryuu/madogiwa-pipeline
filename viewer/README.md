# viewer

`data/traces_2000.jsonl` の prompt / reasoning / answer を閲覧するための Web ビューワ。

## セットアップ・起動

```bash
cd viewer
uv sync
uv run -- uvicorn app:app --reload
```

ブラウザで http://localhost:8000 を開く。

## 機能

- レコード一覧 (ID / subset / prompt 冒頭) をテーブル表示
- subset フィルタ、テキスト検索
- 行クリックで詳細表示: prompt / reasoning / answer を色分け表示
- reasoning は折りたたみ可能
- ← → キーで前後レコード移動、Esc で閉じる
