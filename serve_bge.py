"""本机 BGE：向量和精排。用已经装好的 torch，不额外装包。"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

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
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_DIR, local_files_only=True)
    embed_model = AutoModel.from_pretrained(EMBED_DIR, local_files_only=True)
    embed_model.to(DEVICE).eval()
    rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_DIR, local_files_only=True)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(
        RERANK_DIR, local_files_only=True
    )
    rerank_model.to(DEVICE).eval()
    return embed_tokenizer, embed_model, rerank_tokenizer, rerank_model


print("loading BGE weights", flush=True)
EMBED_TOK, EMBED_MODEL, RERANK_TOK, RERANK_MODEL = load()


def embed(texts: list[str]) -> list[list[float]]:
    batch = EMBED_TOK(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    batch = {key: value.to(DEVICE) for key, value in batch.items()}
    with torch.inference_mode():
        hidden = EMBED_MODEL(**batch).last_hidden_state[:, 0]
        hidden = F.normalize(hidden, p=2, dim=1)
    return hidden.float().cpu().tolist()


def rerank(pairs: list[list[str]]) -> list[float]:
    batch = RERANK_TOK(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    batch = {key: value.to(DEVICE) for key, value in batch.items()}
    with torch.inference_mode():
        logits = RERANK_MODEL(**batch).logits.view(-1)
    return [round(float(score), 4) for score in logits.float().cpu()]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/health":
            self._json(404, {"detail": "没有这个地址"})
            return
        self._json(
            200,
            {
                "status": "ok",
                "embed": f"http://{HOST}:{PORT}/embed",
                "rerank": f"http://{HOST}:{PORT}/rerank",
            },
        )

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"detail": "请求不是 JSON"})
            return
        try:
            if path == "/embed":
                texts = body.get("texts") or []
                if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
                    raise ValueError("texts 需要是非空字符串列表")
                vectors = embed(texts)
                self._json(200, {"model": "BAAI/bge-large-zh-v1.5", "dim": len(vectors[0]), "embeddings": vectors})
                return
            if path == "/rerank":
                pairs = body.get("pairs")
                if pairs is None and body.get("query") and body.get("documents"):
                    pairs = [[body["query"], doc] for doc in body["documents"]]
                if not pairs:
                    raise ValueError("请传 pairs，或 query 加 documents")
                scores = rerank(pairs)
                results = [
                    {"index": i, "text_a": left, "text_b": right, "score": score}
                    for i, ((left, right), score) in enumerate(zip(pairs, scores))
                ]
                results.sort(key=lambda item: item["score"], reverse=True)
                self._json(200, {"model": "BAAI/bge-reranker-large", "results": results})
                return
        except ValueError as exc:
            self._json(400, {"detail": str(exc)})
            return
        except Exception as exc:
            self._json(500, {"detail": f"推理失败: {exc}"})
            return
        self._json(404, {"detail": "没有这个地址"})

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print(f"listening on http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
