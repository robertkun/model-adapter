#!/usr/bin/env python3
"""将 MLX fuse 后的模型整理为 Ollama 可导入的 HuggingFace safetensors 目录。"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

from safetensors.numpy import save_file
import mlx.core as mx


ROOT = Path(__file__).resolve().parent
FUSED_DIR = ROOT / "my_first_model"
BASE_DIR = ROOT / "model" / "Qwen" / "Qwen3.5-2B"
OUT_DIR = ROOT / "ollama_model"


def load_safetensors_as_mx(path: Path) -> dict[str, mx.array]:
    return dict(mx.load(str(path)))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 基座权重（含 vision / mtp）
    base_files = sorted(BASE_DIR.glob("*.safetensors"))
    if not base_files:
        raise FileNotFoundError(f"未找到基座权重: {BASE_DIR}")

    weights: dict[str, mx.array] = {}
    for f in base_files:
        if f.name.endswith(".index.json"):
            continue
        weights.update(load_safetensors_as_mx(f))

    # 微调后的语言模型权重（MLX fuse 使用 language_model.model.*）
    fused = load_safetensors_as_mx(FUSED_DIR / "model.safetensors")
    remapped = 0
    for k, v in fused.items():
        if k.startswith("language_model.model."):
            new_k = "model.language_model." + k[len("language_model.model.") :]
        elif k.startswith("language_model."):
            new_k = "model." + k
        else:
            new_k = k
        weights[new_k] = v
        remapped += 1

    print(f"✅ 已映射微调权重 {remapped} 个张量")
    print(f"✅ 合并后总张量数: {len(weights)}")

    # 转 numpy 保存（safetensors）
    np_weights = {k: v.astype(mx.float16) for k, v in weights.items()}
    # mlx -> numpy via mx
    import numpy as np

    out_tensors = {}
    for k, arr in np_weights.items():
        out_tensors[k] = np.array(arr, copy=False)

    out_path = OUT_DIR / "model.safetensors"
    save_file(out_tensors, str(out_path))
    print(f"✅ 已写入 {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")

    # 配置与分词器：优先用基座完整配置（保留多模态结构，便于 Ollama 转换）
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "configuration.json",
        "merges.txt",
        "vocab.json",
    ]:
        src = BASE_DIR / name
        if not src.exists():
            alt = FUSED_DIR / name
            src = alt if alt.exists() else None
        if src and src.exists():
            shutil.copy2(src, OUT_DIR / name)

    # 确保 architectures 字段存在
    cfg_path = OUT_DIR / "config.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("architectures", ["Qwen3_5ForConditionalGeneration"])
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Modelfile
    modelfile = OUT_DIR / "Modelfile"
    modelfile.write_text(
        """FROM .
TEMPLATE "{{ .Prompt }}"
RENDERER qwen3.5
PARSER qwen3.5
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER num_ctx 4096
SYSTEM 你是小胖，由 Robert 开发的人工智能助手。请用简洁自然的中文回答。
""",
        encoding="utf-8",
    )
    print(f"✅ Modelfile: {modelfile}")
    print("下一步: ollama create xiaopang -f ollama_model/Modelfile")


if __name__ == "__main__":
    main()
