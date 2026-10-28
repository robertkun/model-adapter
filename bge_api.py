"""本地接口：bge-large-zh-v1.5 向量，bge-reranker-large 重排序。"""

from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent
EMBED_DIR = (
    ROOT
    / "bge-large-zh-v1.5"
    / "models"
    / "BAAI--bge-large-zh-v1.5"
    / "snapshots"
    / "master"
)
RERANK_DIR = (
    ROOT
    / "bge-reranker-large"
    / "models"
    / "BAAI--bge-reranker-large"
    / "snapshots"
    / "master"
)
HOST = "127.0.0.1"
PORT = 8765

device = "mps" if torch.backends.mps.is_available() else "cpu"
embedder = SentenceTransformer(str(EMBED_DIR), local_files_only=True, device=device)
reranker = CrossEncoder(str(RERANK_DIR), local_files_only=True, device=device)

app = FastAPI(title="本地 BGE 接口")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    normalize: bool = True
    return_embeddings: bool = False


class RerankRequest(BaseModel):
    pairs: list[tuple[str, str]] | None = None
    query: str | None = None
    documents: list[str] | None = None


def resolve_pairs(body: RerankRequest) -> list[tuple[str, str]]:
    if body.pairs:
        return [(a, b) for a, b in body.pairs]
    if body.query is not None and body.documents:
        return [(body.query, doc) for doc in body.documents]
    raise HTTPException(status_code=400, detail="请传 pairs，或 query 加 documents")


PAGE = ROOT / "static" / "index.html"


class EvalCase(BaseModel):
    id: str = ""
    a: str
    b: str
    label: str


class EvalRequest(BaseModel):
    cases: list[EvalCase] = Field(min_length=1, max_length=2000)
    embedding_threshold: float = 0.92
    rerank_threshold: float = 7


@app.get("/")
def index():
    return FileResponse(PAGE)


@app.get("/health")
def health():
    return {
        "embed": f"http://{HOST}:{PORT}/embed",
        "rerank": f"http://{HOST}:{PORT}/rerank",
        "eval": f"http://{HOST}:{PORT}/eval",
        "eval_qwen": f"http://{HOST}:{PORT}/eval/qwen",
        "eval_bonsai": f"http://{HOST}:{PORT}/eval/bonsai",
    }


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


def bge_encode(texts: list[str]) -> np.ndarray:
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(vectors, dtype=np.float64)
    emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    return emb


def bge_rerank(pairs: list[tuple[str, str]]) -> list[float]:
    logits = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
        activation_fn=torch.nn.Identity(),
    )
    return [round(float(value), 4) for value in np.atleast_1d(np.asarray(logits, dtype=float))]


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

    # embedding 过阈值的全部进入精排，期望 hit 也包含在内。
    entered = [case for case in cases if case["embedding_pred"] == "hit"]
    if entered:
        scores = rerank([(case["a"], case["b"]) for case in entered])
        for case, score in zip(entered, scores):
            case["rerank"] = score
            case["rerank_pred"] = "hit" if score >= rerank_threshold else "miss"


@app.post("/eval")
def evaluate(body: EvalRequest):
    cases = parse_cases(body)
    run_pair_eval(
        cases,
        bge_encode,
        bge_rerank,
        body.embedding_threshold,
        body.rerank_threshold,
    )

    return eval_payload(cases, body.embedding_threshold, body.rerank_threshold)


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


@app.get("/health/qwen")
def qwen_health():
    import qwen_local

    return {
        "ready": qwen_local.ready(),
        "embed": qwen_local.EMBED_REPO,
        "rerank": qwen_local.RERANK_REPO,
        "message": "" if qwen_local.ready() else qwen_local.MISSING,
    }


@app.post("/eval/qwen")
def evaluate_qwen(body: EvalRequest):
    import qwen_local

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


@app.get("/health/bonsai")
def bonsai_health():
    import bonsai_local

    return {
        "ready": bonsai_local.ready(),
        "running": bonsai_local.server_up(),
        "model": bonsai_local.MODEL_NAME,
        "message": "" if bonsai_local.ready() else bonsai_local.MISSING,
    }


@app.post("/eval/bonsai")
def evaluate_bonsai(body: EvalRequest):
    import bonsai_local

    if not bonsai_local.ready():
        raise HTTPException(status_code=503, detail=bonsai_local.MISSING)
    rows = []
    try:
        for item in body.cases:
            label = item.label.strip().lower()
            left, right = item.a.strip(), item.b.strip()
            if label not in {"hit", "miss"}:
                raise HTTPException(status_code=400, detail="标注只能是 hit 或 miss")
            if not left or not right:
                rows.append(
                    {
                        "id": item.id.strip(),
                        "a": left,
                        "b": right,
                        "label": label,
                        "answer": "句子为空",
                        "pred": "miss",
                        "consistent": label == "miss",
                    }
                )
                continue
            answer, pred = bonsai_local.judge(left, right)
            rows.append(
                {
                    "id": item.id.strip(),
                    "a": left,
                    "b": right,
                    "label": label,
                    "answer": answer,
                    "pred": pred,
                    "consistent": pred == label,
                }
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Bonsai 判断失败: {exc}") from exc
    return {"rows": rows}


@app.post("/embed")
def embed(body: EmbedRequest):
    vectors = embedder.encode(
        body.texts,
        normalize_embeddings=body.normalize,
        show_progress_bar=False,
    )
    arr = np.asarray(vectors, dtype=float)
    similarities = []
    for i in range(len(body.texts)):
        for j in range(i + 1, len(body.texts)):
            similarities.append(
                {
                    "a": body.texts[i],
                    "b": body.texts[j],
                    "score": round(float(np.dot(arr[i], arr[j])), 4),
                }
            )
    payload = {
        "model": "BAAI/bge-large-zh-v1.5",
        "dim": int(arr.shape[-1]),
        "similarities": similarities,
    }
    if body.return_embeddings:
        payload["embeddings"] = arr.tolist()
    return payload


@app.post("/rerank")
def rerank(body: RerankRequest):
    pairs = resolve_pairs(body)
    logits = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
        activation_fn=torch.nn.Identity(),
    )
    scores = np.atleast_1d(np.asarray(logits, dtype=float))
    results = []
    for i, ((left, right), score) in enumerate(zip(pairs, scores)):
        results.append(
            {
                "index": i,
                "text_a": left,
                "text_b": right,
                "score": float(score),
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return {"model": "BAAI/bge-reranker-large", "results": results}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
