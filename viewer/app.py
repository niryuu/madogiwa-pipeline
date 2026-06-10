import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

DATA_PATH = Path(__file__).parent.parent / "data" / "traces_2000.jsonl"
ANNOTATIONS_PATH = Path(__file__).parent.parent / "data" / "annotations.jsonl"

# --- data loading ---

records: list[dict] = []
subsets: list[str] = []
annotations: dict[str, dict] = {}  # id -> {"annotation": str, "created_at": str}


def _extract_text(content) -> str:
    """messages の content フィールドからテキストを取り出す。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    if isinstance(content, dict):
        return content.get("text", json.dumps(content, ensure_ascii=False))
    return str(content)


def _parse_record(raw: dict) -> dict:
    """生レコードを表示用に整形する。"""
    prompts = []
    analysis = ""
    final = ""

    for msg in raw.get("messages", []):
        role = msg.get("role", "")
        if role == "user":
            prompts.append(_extract_text(msg.get("content", "")))
        elif role == "assistant":
            channel = msg.get("channel", "")
            text = _extract_text(msg.get("content", ""))
            if channel == "analysis":
                analysis = text
            elif channel == "final":
                final = text

    return {
        "id": raw.get("id", ""),
        "subset": raw.get("subset", ""),
        "split": raw.get("split", ""),
        "prompt": "\n\n---\n\n".join(prompts),
        "prompt_short": prompts[0][:120] if prompts else "",
        "reasoning": analysis,
        "answer": final,
    }


def load_data():
    global records, subsets
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(_parse_record(json.loads(line)))
    subsets = sorted(set(r["subset"] for r in records))


def load_annotations():
    global annotations
    if ANNOTATIONS_PATH.exists():
        with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    annotations[entry["id"]] = {
                        "annotation": entry["annotation"],
                        "created_at": entry.get("created_at", ""),
                    }


def save_annotations():
    with open(ANNOTATIONS_PATH, "w", encoding="utf-8") as f:
        for record_id, data in annotations.items():
            entry = {"id": record_id, **data}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


load_data()
load_annotations()

# --- routes ---


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"subsets": subsets, "total": len(records)},
    )


@app.get("/api/records")
async def api_records(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    subset: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    annotated: Optional[str] = Query(None),  # "yes", "no", or None
):
    filtered = records
    if subset:
        filtered = [r for r in filtered if r["subset"] == subset]
    if q:
        q_lower = q.lower()
        filtered = [
            r for r in filtered
            if q_lower in r["prompt"].lower()
            or q_lower in r["answer"].lower()
            or q_lower in r["id"].lower()
        ]
    if annotated == "yes":
        filtered = [r for r in filtered if r["id"] in annotations]
    elif annotated == "no":
        filtered = [r for r in filtered if r["id"] not in annotations]

    total = len(filtered)
    start = (page - 1) * per_page
    end = start + per_page
    page_records = []
    for r in filtered[start:end]:
        rec = {**r}
        ann = annotations.get(r["id"])
        rec["annotation"] = ann["annotation"] if ann else ""
        page_records.append(rec)

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
        "records": page_records,
    }


class AnnotationBody(BaseModel):
    annotation: str


@app.put("/api/annotations/{record_id}")
async def put_annotation(record_id: str, body: AnnotationBody):
    annotations[record_id] = {
        "annotation": body.annotation,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_annotations()
    return {"ok": True, "id": record_id}


@app.delete("/api/annotations/{record_id}")
async def delete_annotation(record_id: str):
    annotations.pop(record_id, None)
    save_annotations()
    return {"ok": True, "id": record_id}
