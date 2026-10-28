"""用本机 Bonsai 2 27B 判断两句话是不是同一个意思。"""

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = (
    ROOT
    / "deploy"
    / "bonsai2-mac"
    / "models"
    / "Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf"
)
LOG = ROOT / "deploy" / "bonsai2-mac" / "server.log"
PID = ROOT / "deploy" / "bonsai2-mac" / "server.pid"
HOST = "127.0.0.1"
PORT = 8091
MODEL_NAME = "Ternary Bonsai 2 27B"
MISSING = "还没有本机 Bonsai 权重。先在 deploy/bonsai2-mac 执行 ./download.sh"

SYSTEM = (
    "判断句子A和句子B是不是同一个意思。"
    "同一个意思只回答 hit，不是同一个意思只回答 miss。"
    "不要解释。"
)

_lock = threading.Lock()
_proc: subprocess.Popen | None = None


def ready() -> bool:
    return MODEL.is_file() and MODEL.stat().st_size >= 7_000_000_000


def server_up() -> bool:
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def ensure_server() -> None:
    if not ready():
        raise RuntimeError(MISSING)
    if server_up():
        return
    with _lock:
        if server_up():
            return
        global _proc
        LOG.parent.mkdir(parents=True, exist_ok=True)
        log = open(LOG, "ab")
        _proc = subprocess.Popen(
            [
                "llama-server",
                "-m",
                str(MODEL),
                "--host",
                HOST,
                "--port",
                str(PORT),
                "-ngl",
                "99",
                "-t",
                "10",
                "-c",
                "4096",
                "-np",
                "1",
                "-fa",
                "on",
                "--reasoning",
                "off",
                "--temp",
                "0",
                "--top-k",
                "20",
                "--top-p",
                "0.95",
            ],
            stdout=log,
            stderr=log,
        )
        PID.write_text(str(_proc.pid))
        deadline = time.time() + 180
        while time.time() < deadline:
            if server_up():
                return
            if _proc.poll() is not None:
                tail = LOG.read_text(errors="replace")[-800:]
                raise RuntimeError(f"模型服务退出了。{tail}")
            time.sleep(0.5)
        raise RuntimeError("模型服务启动超时")


def judge(left: str, right: str) -> tuple[str, str | None]:
    ensure_server()
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"句子A：{left}\n句子B：{right}"},
        ],
        "temperature": 0,
        "max_tokens": 16,
    }
    request = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.load(response)
    message = body["choices"][0]["message"]
    answer = (message.get("content") or "").strip()
    return answer, parse_verdict(answer)


def parse_verdict(text: str) -> str | None:
    cleaned = text
    for mark in ("[End thinking]", "</think>"):
        if mark in cleaned:
            cleaned = cleaned.split(mark)[-1]
    found = []
    lower = cleaned.lower()
    start = 0
    while True:
        hit_at = lower.find("hit", start)
        miss_at = lower.find("miss", start)
        positions = [pos for pos in (hit_at, miss_at) if pos >= 0]
        if not positions:
            break
        pos = min(positions)
        word = "hit" if pos == hit_at else "miss"
        found.append(word)
        start = pos + len(word)
    return found[-1] if found else None
