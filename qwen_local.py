"""本地 Qwen3-Embedding-8B 与 Qwen3-Reranker-8B。用完即释放，避免和 BGE 同时占满内存。"""

import gc
import os
from pathlib import Path

# 这台 CPU 没有 bfloat16 指令。线程数要在 import torch 之前定好，否则只会占满 1 个核。
_THREADS = min(16, os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_THREADS))

import numpy as np
import torch

torch.set_num_threads(_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parent
EMBED_DIR = ROOT / "qwen" / "Qwen3-Embedding-8B"
RERANK_DIR = ROOT / "qwen" / "Qwen3-Reranker-8B"
EMBED_REPO = "Qwen/Qwen3-Embedding-8B"
RERANK_REPO = "Qwen/Qwen3-Reranker-8B"

MISSING = (
    "本地还没有 Qwen3-Embedding-8B 和 Qwen3-Reranker-8B。"
    "两个模型权重大约 32GB，需要先下载到 qwen 目录后才能测试。"
)


def ready() -> bool:
    return (EMBED_DIR / "config.json").is_file() and (RERANK_DIR / "config.json").is_file()


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype() -> torch.dtype:
    # 旧 Xeon 没有 bfloat16 指令，用 bfloat16 会退化成单核模拟，所以 CPU 用 float32。
    if _device() == "cpu":
        return torch.float32
    return torch.float16


def _release(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(
        f"qwen embed device={_device()} dtype={_dtype()} threads={torch.get_num_threads()} texts={len(texts)}",
        flush=True,
    )

    model = SentenceTransformer(
        str(EMBED_DIR),
        local_files_only=True,
        device=_device(),
        model_kwargs={"torch_dtype": _dtype()},
    )
    try:
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=1 if _device() == "cpu" else 4,
            show_progress_bar=False,
        )
    finally:
        _release(model)
    emb = np.asarray(vectors, dtype=np.float64)
    emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    return emb


def rerank(pairs: list[tuple[str, str]]) -> list[float]:
    """返回「两句同义」的概率，范围 0 到 1。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(
        f"qwen rerank device={_device()} dtype={_dtype()} threads={torch.get_num_threads()} pairs={len(pairs)}",
        flush=True,
    )

    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(
        str(RERANK_DIR),
        padding_side="left",
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(RERANK_DIR),
        torch_dtype=_dtype(),
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if device != "cpu":
        model = model.to(device)
    model.eval()
    token_true = tokenizer.convert_tokens_to_ids("yes")
    token_false = tokenizer.convert_tokens_to_ids("no")
    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    instruction = "Judge whether the Document has the same meaning as the Query."
    max_length = 2048
    room = max_length - len(prefix_ids) - len(suffix_ids)
    batch_size = 4 if device == "cpu" else 8
    scores: list[float] = []
    try:
        for start in range(0, len(pairs), batch_size):
            batch_ids = []
            for left, right in pairs[start : start + batch_size]:
                body = f"<Instruct>: {instruction}\n<Query>: {left}\n<Document>: {right}"
                body_ids = tokenizer.encode(body, add_special_tokens=False)
                batch_ids.append(prefix_ids + body_ids[:room] + suffix_ids)
            inputs = tokenizer.pad({"input_ids": batch_ids}, padding=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs).logits[:, -1, :]
            pair_logits = torch.stack([logits[:, token_false], logits[:, token_true]], dim=1)
            probs = torch.softmax(pair_logits.float(), dim=1)[:, 1]
            scores.extend(round(float(prob), 4) for prob in probs)
    finally:
        _release(model)
    return scores
