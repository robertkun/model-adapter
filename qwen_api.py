"""服务器上的 Qwen 测试页。只加载 Qwen3-Embedding-8B 和 Qwen3-Reranker-8B。"""

from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import qwen_local

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "static" / "index.html"
HOST = "0.0.0.0"
PORT = 8765

app = FastAPI(title="Qwen 测试")


class EvalCase(BaseModel):
    id: str = ""
    a: str
    b: str
    label: str


class EvalRequest(BaseModel):
    cases: list[EvalCase] = Field(min_length=1, max_length=2000)
    embedding_threshold: float = 0.6
    rerank_threshold: float = 0.5


@app.get("/")
def index():
    return FileResponse(PAGE)


@app.get("/health/qwen")
def qwen_health():
    return {
        "ready": qwen_local.ready(),
        "embed": qwen_local.EMBED_REPO,
        "rerank": qwen_local.RERANK_REPO,
        "embed_dir": str(qwen_local.EMBED_DIR),
        "rerank_dir": str(qwen_local.RERANK_DIR),
        "message": "" if qwen_local.ready() else qwen_local.MISSING,
    }


@app.post("/eval")
def evaluate_baai(body: EvalRequest):
    raise HTTPException(status_code=503, detail="这台服务器只部署了 Qwen，请使用 QWEN 标签")


def parse_cases(body: EvalRequest) -> list[dict]:
    cases = []
    for item in body.cases:
        label = item.label.strip().lower()
        left, right = item.a.strip(), item.b.strip()
        if label not in {"hit", "miss"}:
            raise HTTPException(status_code=400, detail="标注只能是 hit 或 miss")
        if not left or not right:
            raise HTTPException(status_code=400, detail="句子 A 和句子 B 都不能为空")
        cases.append({"id": item.id.strip(), "a": left, "b": right, "label": label})
    return cases


def run_pair_eval(cases, encode, rerank, embedding_threshold: float, rerank_threshold: float):
    texts: list[str] = []
    index: dict[str, int] = {}
    for case in cases:
        for text in (case["a"], case["b"]):
            if text not in index:
                index[text] = len(texts)
                texts.append(text)

    emb = encode(texts)
    sims = emb @ emb.T
    for case in cases:
        ia, ib = index[case["a"]], index[case["b"]]
        if case["a"] == case["b"]:
            cosine, rank = 1.0, 1
        else:
            row = sims[ia].copy()
            row[ia] = -np.inf
            cosine = float(row[ib])
            rank = 1 + int(np.sum(row > cosine + 1e-12))
        embedding = round(cosine, 4)
        case["embedding"] = embedding
        case["rank"] = rank
        case["embedding_pred"] = "hit" if embedding >= embedding_threshold else "miss"

    entered = [case for case in cases if case["embedding_pred"] == "hit"]
    if entered:
        scores = rerank([(case["a"], case["b"]) for case in entered])
        for case, score in zip(entered, scores):
            case["rerank"] = score
            case["rerank_pred"] = "hit" if score >= rerank_threshold else "miss"


def eval_payload(cases, embedding_threshold: float, rerank_threshold: float):
    rows = []
    for case in cases:
        entered_rerank = case["embedding_pred"] == "hit"
        final_pred = case["rerank_pred"] if entered_rerank else "miss"
        rows.append(
            {
                "id": case["id"],
                "a": case["a"],
                "b": case["b"],
                "label": case["label"],
                "embedding": case["embedding"],
                "embedding_pred": case["embedding_pred"],
                "rank": case["rank"],
                "entered_rerank": entered_rerank,
                "rerank": case.get("rerank"),
                "rerank_pred": case.get("rerank_pred"),
                "final_pred": final_pred,
                "consistent": final_pred == case["label"],
            }
        )
    expected = [row for row in rows if row["label"] == "hit"]
    recalled = [row for row in rows if row["entered_rerank"]]
    passed = [row for row in recalled if row["rerank_pred"] == "hit"]
    return {
        "embedding_threshold": embedding_threshold,
        "rerank_threshold": rerank_threshold,
        "summary": {
            "total": len(rows),
            "expected_hit": len(expected),
            "recalled": len(recalled),
            "recalled_hit": sum(row["label"] == "hit" for row in recalled),
            "rerank_passed": len(passed),
            "rerank_passed_hit": sum(row["label"] == "hit" for row in passed),
            "hit_recalled": sum(row["entered_rerank"] for row in expected),
            "hit_rank5": sum(row["rank"] <= 5 for row in expected),
            "hit_rerank_passed": sum(
                row["entered_rerank"] and row["rerank_pred"] == "hit" for row in expected
            ),
        },
        "rows": rows,
    }


@app.post("/eval/qwen")
def evaluate_qwen(body: EvalRequest):
    if not qwen_local.ready():
        raise HTTPException(status_code=503, detail=qwen_local.MISSING)
    cases = parse_cases(body)
    try:
        run_pair_eval(
            cases,
            qwen_local.embed,
            qwen_local.rerank,
            body.embedding_threshold,
            body.rerank_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Qwen 推理失败: {exc}") from exc
    return eval_payload(cases, body.embedding_threshold, body.rerank_threshold)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
