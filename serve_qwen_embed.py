"""本机 Qwen3-Embedding-8B。只提供向量，不加载精排。"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "qwen" / "Qwen3-Embedding-8B"
HOST = "127.0.0.1"
PORT = 8093
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32


def last_token_pool(hidden, mask):
    if int(mask[:, -1].sum()) == mask.shape[0]:
        return hidden[:, -1]
    lengths = mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


def load():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, padding_side="left"
    )
    model = AutoModel.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=DTYPE
    )
    model.to(DEVICE).eval()
    return tokenizer, model


print("loading Qwen3-Embedding-8B", flush=True)
TOKENIZER, MODEL = load()


def embed(texts: list[str]) -> list[list[float]]:
    batch = TOKENIZER(
        texts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )
    batch = {key: value.to(DEVICE) for key, value in batch.items()}
    with torch.inference_mode():
        hidden = MODEL(**batch).last_hidden_state
        pooled = last_token_pool(hidden, batch["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.float().cpu().tolist()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/health":
            self._json(404, {"detail": "没有这个地址"})
            return
        self._json(
            200,
            {"status": "ok", "embed": f"http://{HOST}:{PORT}/embed", "model": "Qwen/Qwen3-Embedding-8B"},
        )

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/embed":
            self._json(404, {"detail": "没有这个地址"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            texts = body.get("texts") or []
            if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
                raise ValueError("texts 需要是非空字符串列表")
            vectors = embed(texts)
        except ValueError as exc:
            self._json(400, {"detail": str(exc)})
            return
        except Exception as exc:
            self._json(500, {"detail": f"推理失败: {exc}"})
            return
        self._json(
            200,
            {"model": "Qwen/Qwen3-Embedding-8B", "dim": len(vectors[0]), "embeddings": vectors},
        )

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
