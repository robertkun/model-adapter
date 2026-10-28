"""用 bge-large-zh-v1.5 验证句对，对照 hit/miss。"""

import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
CASES_FILE = ROOT / "docs" / "验证.md"
MODEL_DIR = (
    ROOT
    / "bge-large-zh-v1.5"
    / "models"
    / "BAAI--bge-large-zh-v1.5"
    / "snapshots"
    / "master"
)

PAIR_RE = re.compile(
    r'\{"(?P<id>[^"]+)", "(?P<a>(?:\\.|[^"\\])*)", "(?P<b>(?:\\.|[^"\\])*)", "(?P<label>hit|miss)"\}'
)


def load_cases(path: Path):
    text = path.read_text(encoding="utf-8")
    cases = []
    for m in PAIR_RE.finditer(text):
        cases.append(
            (
                m.group("id"),
                m.group("a").encode("utf-8").decode("unicode_escape")
                if "\\" in m.group("a")
                else m.group("a"),
                m.group("b").encode("utf-8").decode("unicode_escape")
                if "\\" in m.group("b")
                else m.group("b"),
                m.group("label"),
            )
        )
    if not cases:
        raise SystemExit(f"没有解析到句对: {path}")
    return cases


def main() -> None:
    cases = load_cases(CASES_FILE)
    texts = []
    index = {}
    for _, a, b, _ in cases:
        for t in (a, b):
            if t not in index:
                index[t] = len(texts)
                texts.append(t)

    model = SentenceTransformer(str(MODEL_DIR), local_files_only=True)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    rows = []
    for cid, a, b, label in cases:
        score = float(np.dot(vectors[index[a]], vectors[index[b]]))
        rows.append((cid, a, b, label, score))

    # 找让 hit >= t 且 miss < t 的最佳阈值
    scores = sorted({round(r[4], 4) for r in rows})
    candidates = [0.0, 1.01]
    for s in scores:
        candidates.append(s)
        candidates.append(round(s + 1e-6, 6))
    best = None
    for t in candidates:
        ok = 0
        for _, _, _, label, score in rows:
            pred = "hit" if score >= t else "miss"
            if pred == label:
                ok += 1
        acc = ok / len(rows)
        if best is None or acc > best[0] or (acc == best[0] and t > best[1]):
            best = (acc, t, ok)

    acc, threshold, ok = best
    print(f"句对数: {len(rows)}")
    print(f"向量维度: {vectors.shape[1]}")
    print(f"最佳阈值: {threshold:.4f}  一致 {ok}/{len(rows)}  准确率 {acc:.1%}")
    print()

    mismatches = []
    for cid, a, b, label, score in rows:
        pred = "hit" if score >= threshold else "miss"
        mark = "OK" if pred == label else "DIFF"
        if pred != label:
            mismatches.append((cid, label, pred, score, a, b))
        print(f"{mark:4} {score:.4f}  期望{label:4} 预测{pred:4}  {cid}")
        print(f"         A: {a if a else '<空>'}")
        print(f"         B: {b if b else '<空>'}")

    print()
    print(f"不一致 {len(mismatches)} 条:")
    for cid, label, pred, score, a, b in mismatches:
        print(f"  {cid}  score={score:.4f}  期望{label} 实际更像{pred}")
        print(f"    A: {a if a else '<空>'}")
        print(f"    B: {b if b else '<空>'}")


if __name__ == "__main__":
    main()
