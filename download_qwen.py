# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys

if sys.version_info[0] < 3:
    script = os.path.abspath(__file__)
    for name in ("python3", "python3.12", "python3.11", "python3.10", "python3.9", "python3.8", "python3.7", "python3.6"):
        for folder in os.environ.get("PATH", "").split(os.pathsep):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                os.execv(path, [path, script] + sys.argv[1:])
    sys.stderr.write("当前 python 是 Python 2，且找不到 python3。请先安装 Python 3。\n")
    sys.exit(1)

if sys.version_info < (3, 8):
    sys.exit("需要 Python 3.8 或更高版本，当前是 Python %s" % sys.version.split()[0])

import shutil
from pathlib import Path

try:
    from modelscope import snapshot_download
except ImportError:
    sys.exit("缺少 modelscope。先执行: python3 -m pip install modelscope")

ROOT = Path(__file__).resolve().parent
NEED_BYTES = 34 * 1024 ** 3
MODELS = (
    ("Qwen/Qwen3-Embedding-8B", ROOT / "qwen" / "Qwen3-Embedding-8B"),
    ("Qwen/Qwen3-Reranker-8B", ROOT / "qwen" / "Qwen3-Reranker-8B"),
)


def main():
    free = shutil.disk_usage(str(ROOT)).free
    if free < NEED_BYTES:
        gb = free / float(1024 ** 3)
        sys.exit("磁盘剩余 %.1f GB，两个 8B 模型大约需要 32 GB，空间不够，已停止下载。" % gb)
    for repo, dest in MODELS:
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / "config.json").is_file():
            print("已存在: %s" % dest)
            continue
        print("下载 %s -> %s" % (repo, dest))
        snapshot_download(repo, local_dir=str(dest))
        print("完成: %s" % dest)


if __name__ == "__main__":
    main()
