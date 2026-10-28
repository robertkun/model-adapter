"""用 BAAI/bge-large-zh-v1.5 计算中文句子的语义相似度。"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_DIR = (
    Path(__file__).resolve().parent
    / "bge-large-zh-v1.5"
    / "models"
    / "BAAI--bge-large-zh-v1.5"
    / "snapshots"
    / "master"
)

# 句对：预期从高到低
PAIRS = [
    ("今天天气真好，适合出去散步", "今日阳光明媚，很适合户外走走"),
    ("如何办理护照", "护照申请需要哪些材料"),
    ("苹果公司发布了新款手机", "水果店的苹果很新鲜"),
    ("机器学习模型如何训练", "今晚食堂有什么菜"),
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    model = SentenceTransformer(str(MODEL_DIR), local_files_only=True)
    texts = [t for pair in PAIRS for t in pair]
    vectors = model.encode(texts, normalize_embeddings=True)

    print(f"模型: {MODEL_DIR.name}")
    print(f"向量维度: {vectors.shape[1]}")
    print()
    for i, (left, right) in enumerate(PAIRS):
        score = cosine(vectors[i * 2], vectors[i * 2 + 1])
        print(f"{score:.4f}  {left}")
        print(f"        {right}")
        print()


if __name__ == "__main__":
    main()
