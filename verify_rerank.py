"""用 bge-reranker-large 对句对打分，对照 hit/miss。"""

import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent
CASES_FILE = ROOT / "docs" / "验证.md"
MODEL_DIR = (
    ROOT
    / "bge-reranker-large"
    / "models"
    / "BAAI--bge-reranker-large"
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
        def dec(s: str) -> str:
            return s.encode("utf-8").decode("unicode_escape") if "\\" in s else s

        cases.append((m.group("id"), dec(m.group("a")), dec(m.group("b")), m.group("label")))
    if not cases:
        raise SystemExit(f"没有解析到句对: {path}")
    return cases


def best_threshold(rows):
    scores = sorted({r[4] for r in rows})
    candidates = [-1e9, 1e9]
    for s in scores:
        candidates.append(s)
        candidates.append(np.nextafter(s, 1e9))
    best = None
    for t in candidates:
        ok = sum(("hit" if score >= t else "miss") == label for _, _, _, label, score in rows)
        acc = ok / len(rows)
        if best is None or acc > best[0]:
            best = (acc, float(t), ok)
    return best


def main() -> None:
    cases = load_cases(CASES_FILE)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    model = CrossEncoder(str(MODEL_DIR), local_files_only=True, device=device)
    pairs = [(a, b) for _, a, b, _ in cases]
    # CrossEncoder 默认已经做了 sigmoid，这里取原始 logit，避免分数挤在 0.73。
    model.model.eval()
    logits = model.predict(
        pairs,
        batch_size=16,
        show_progress_bar=True,
        activation_fn=torch.nn.Identity(),
    )

    rows = [
        (cid, a, b, label, float(logit))
        for (cid, a, b, label), logit in zip(cases, logits)
    ]
    acc, threshold, ok = best_threshold(rows)
    misses_as_hit = 0
    hits_as_miss = 0
    mismatches = []
    print(f"句对数: {len(rows)}")
    print(f"最佳阈值: {threshold:.4f}  一致 {ok}/{len(rows)}  准确率 {acc:.1%}")
    print()
    for cid, a, b, label, score in rows:
        pred = "hit" if score >= threshold else "miss"
        mark = "OK" if pred == label else "DIFF"
        if pred != label:
            mismatches.append((cid, label, pred, score, a, b))
            if label == "miss":
                misses_as_hit += 1
            else:
                hits_as_miss += 1
        print(f"{mark:4} {score:.4f}  期望{label:4} 预测{pred:4}  {cid}")
        print(f"         A: {a if a else '<空>'}")
        print(f"         B: {b if b else '<空>'}")

    print()
    print(f"不一致 {len(mismatches)} 条：期望 miss 却判 hit {misses_as_hit}，期望 hit 却判 miss {hits_as_miss}")
    for cid, label, pred, score, a, b in mismatches:
        print(f"  {cid}  score={score:.4f}  期望{label} 预测{pred}")
        print(f"    A: {a if a else '<空>'}")
        print(f"    B: {b if b else '<空>'}")


if __name__ == "__main__":
    main()
