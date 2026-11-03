"""本机模型管理页。只探测端口和权重，不把模型读进内存。"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "static" / "models.html"
HOST = "127.0.0.1"
PORT = 8090

BONSAI_MODEL = (
    ROOT
    / "deploy"
    / "bonsai2-mac"
    / "models"
    / "Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf"
)
ASR_DIR = ROOT / "deploy" / "qwen3-asr" / "models" / "Qwen3-ASR-1.7B"
ASR_BIN = ROOT / "deploy" / "qwen3-asr" / ".venv" / "bin" / "mlx-qwen3-asr"
BGE_EMBED = (
    ROOT
    / "bge-large-zh-v1.5"
    / "models"
    / "BAAI--bge-large-zh-v1.5"
    / "snapshots"
    / "master"
    / "pytorch_model.bin"
)
BGE_RERANK = (
    ROOT
    / "bge-reranker-large"
    / "models"
    / "BAAI--bge-reranker-large"
    / "snapshots"
    / "master"
    / "pytorch_model.bin"
)
QWEN_EMBED = ROOT / "qwen" / "Qwen3-Embedding-8B" / "config.json"
QWEN_RERANK = ROOT / "qwen" / "Qwen3-Reranker-8B" / "config.json"
PYTHON = "/opt/anaconda3/bin/python"
QWEN_PYTHON = ROOT / "deploy" / "qwen-embed" / ".venv" / "bin" / "python"


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((HOST, port)) == 0


def http_json(url: str, timeout: float = 1.5, headers: dict | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def file_ready(path: Path, min_bytes: int = 1) -> bool:
    return path.is_file() and path.stat().st_size >= min_bytes


def listener_pids(port: int) -> list[int]:
    try:
        output = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    pids = []
    for line in output.split():
        if line.isdigit():
            pids.append(int(line))
    return pids


def process_command(pid: int) -> str:
    try:
        return subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def bonsai_state() -> dict:
    weights = file_ready(BONSAI_MODEL, 7_000_000_000)
    body = http_json(f"http://{HOST}:8091/v1/models") if port_open(8091) else None
    model_id = ""
    if body and body.get("data"):
        model_id = body["data"][0].get("id") or ""
    running = bool(model_id)
    return {
        "id": "bonsai-2-27b",
        "name": "Bonsai 2 27B",
        "task": "对话",
        "weights": weights,
        "running": running,
        "can_start": weights and not running,
        "can_stop": running,
        "endpoint": f"http://{HOST}:8091/v1",
        "model_id": model_id or "bonsai-2-27b",
        "auth": "",
        "detail": "OpenAI 兼容对话。Metal，上下文 32768，思考关闭。不用 API Key。",
        "block": "" if weights else "缺少 Q2_0 权重。",
    }


def asr_state() -> dict:
    weights = (ASR_DIR / "config.json").is_file() and ASR_BIN.is_file()
    body = http_json(f"http://{HOST}:8092/health") if port_open(8092) else None
    running = bool(body and body.get("status") == "ok")
    return {
        "id": "qwen3-asr-1.7b",
        "name": "Qwen3-ASR 1.7B",
        "task": "转写",
        "weights": weights,
        "running": running,
        "can_start": weights and not running,
        "can_stop": running,
        "endpoint": f"http://{HOST}:8092/v1",
        "model_id": "Qwen/Qwen3-ASR-1.7B",
        "auth": "local",
        "detail": "OpenAI 兼容语音转写。float16，MLX。请求头带 Authorization: Bearer local。",
        "block": "" if weights else "缺少权重或推理程序。",
    }


def bge_state() -> dict:
    embed = file_ready(BGE_EMBED, 1_000_000)
    rerank = file_ready(BGE_RERANK, 1_000_000)
    serving = http_json(f"http://{HOST}:8765/health") if port_open(8765) else None
    running = bool(serving and serving.get("embed"))
    ready = embed and rerank
    return {
        "id": "bge",
        "name": "BGE 向量 + 精排",
        "task": "向量 / 精排",
        "weights": ready,
        "running": running,
        "can_start": ready and not running,
        "can_stop": running,
        "endpoint": f"http://{HOST}:8765",
        "model_id": "BAAI/bge-large-zh-v1.5 · BAAI/bge-reranker-large",
        "auth": "",
        "detail": "启动后提供 /embed 和 /rerank。第一次加载权重大约需要一两分钟。",
        "block": "" if ready else "向量或精排权重不完整。",
    }


def qwen_embed_state() -> dict:
    embed = QWEN_EMBED.is_file()
    rerank = QWEN_RERANK.is_file()
    body = http_json(f"http://{HOST}:8093/health") if port_open(8093) else None
    running = bool(body and body.get("status") == "ok")
    if embed and rerank:
        detail = "启动后在 8093 提供 /embed。精排权重也在，这个按钮只拉起向量。"
    elif embed:
        detail = "启动后在 8093 提供 /embed。权重大约 14GB，和正在跑的对话、转写一起开会比较吃内存。精排还没下载。"
    else:
        detail = "缺少向量权重。"
    return {
        "id": "qwen3-embedding-8b",
        "name": "Qwen3-Embedding-8B",
        "task": "向量",
        "weights": embed,
        "running": running,
        "can_start": embed and not running,
        "can_stop": running,
        "endpoint": f"http://{HOST}:8093",
        "model_id": "Qwen/Qwen3-Embedding-8B",
        "auth": "",
        "detail": detail,
        "block": "" if embed else "缺少向量权重。",
    }


def snapshot() -> dict:
    models = [bonsai_state(), asr_state(), bge_state(), qwen_embed_state()]
    ollama = http_json(f"http://{HOST}:11434/api/ps") if port_open(11434) else None
    loaded = []
    if ollama and isinstance(ollama.get("models"), list):
        loaded = [item.get("name") or item.get("model") or "" for item in ollama["models"]]
    return {
        "checked_at": time.strftime("%H:%M:%S"),
        "running": [item for item in models if item["running"]],
        "available": [item for item in models if item["weights"] and not item["running"]],
        "ollama": {
            "up": ollama is not None,
            "loaded": [name for name in loaded if name],
        },
    }


def start_model(model_id: str) -> None:
    if model_id == "bonsai-2-27b":
        if bonsai_state()["running"] or port_open(8091):
            return
        if not file_ready(BONSAI_MODEL, 7_000_000_000):
            raise RuntimeError("缺少 Bonsai 权重")
        log_path = ROOT / "deploy" / "bonsai2-mac" / "server.log"
        launch(
            [
                "/opt/homebrew/bin/llama-server",
                "-m",
                str(BONSAI_MODEL),
                "--alias",
                "bonsai-2-27b",
                "--host",
                HOST,
                "--port",
                "8091",
                "-ngl",
                "99",
                "-t",
                "10",
                "-c",
                "32768",
                "-np",
                "1",
                "-fa",
                "on",
                "--reasoning",
                "off",
                "--temp",
                "0.7",
                "--top-p",
                "0.8",
                "--top-k",
                "20",
                "--min-p",
                "0",
            ],
            log_path,
            8091,
        )
        return
    if model_id == "qwen3-asr-1.7b":
        if asr_state()["running"] or port_open(8092):
            return
        if not ASR_BIN.is_file() or not (ASR_DIR / "config.json").is_file():
            raise RuntimeError("缺少 Qwen3-ASR 权重或推理程序")
        log_path = ROOT / "deploy" / "qwen3-asr" / "server.log"
        launch(
            [
                str(ASR_BIN),
                "serve",
                "--host",
                HOST,
                "--port",
                "8092",
                "--api-key",
                "local",
                "--model",
                str(ASR_DIR),
                "--dtype",
                "float16",
            ],
            log_path,
            8092,
        )
        return
    if model_id == "bge":
        if bge_state()["running"]:
            return
        if port_open(8765):
            raise RuntimeError("8765 已被其他程序占用")
        if not (file_ready(BGE_EMBED, 1_000_000) and file_ready(BGE_RERANK, 1_000_000)):
            raise RuntimeError("BGE 权重不完整")
        launch(
            [PYTHON, str(ROOT / "serve_bge.py")],
            ROOT / "deploy" / "bge" / "server.log",
            8765,
            wait_seconds=180,
        )
        return
    if model_id == "qwen3-embedding-8b":
        if qwen_embed_state()["running"]:
            return
        if port_open(8093):
            raise RuntimeError("8093 已被其他程序占用")
        if not QWEN_EMBED.is_file():
            raise RuntimeError("缺少 Qwen3-Embedding-8B 权重")
        python = str(QWEN_PYTHON if QWEN_PYTHON.is_file() else Path(PYTHON))
        launch(
            [python, str(ROOT / "serve_qwen_embed.py")],
            ROOT / "deploy" / "qwen-embed" / "server.log",
            8093,
            wait_seconds=300,
        )
        return
    raise RuntimeError("这个模型不能从管理页启动")


def launch(command: list[str], log_path: Path, port: int, wait_seconds: int = 60) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        subprocess.Popen(
            command,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(ROOT),
        )
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if port_open(port):
            return
        time.sleep(0.4)
    raise RuntimeError(f"端口 {port} 在 {wait_seconds} 秒内没有开始监听，查看 {log_path.name}")


def stop_model(model_id: str) -> None:
    expected = {
        "bonsai-2-27b": (8091, "llama-server"),
        "qwen3-asr-1.7b": (8092, "mlx-qwen3-asr"),
        "bge": (8765, "serve_bge.py"),
        "qwen3-embedding-8b": (8093, "serve_qwen_embed.py"),
    }
    if model_id not in expected:
        raise RuntimeError("这个模型不能从管理页停止")
    port, marker = expected[model_id]
    pids = [pid for pid in listener_pids(port) if marker in process_command(pid)]
    if not pids:
        return
    for pid in pids:
        subprocess.run(["kill", str(pid)], check=False)
    deadline = time.time() + 8
    while time.time() < deadline and port_open(port):
        time.sleep(0.2)
    if port_open(port):
        for pid in pids:
            subprocess.run(["kill", "-9", str(pid)], check=False)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._file(PAGE, "text/html; charset=utf-8")
            return
        if path == "/api/models":
            self._json(200, snapshot())
            return
        self._json(404, {"detail": "没有这个地址"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        prefix = "/api/models/"
        if not path.startswith(prefix):
            self._json(404, {"detail": "没有这个地址"})
            return
        rest = path[len(prefix) :].strip("/")
        model_id, _, action = rest.partition("/")
        try:
            if action == "start":
                start_model(model_id)
            elif action == "stop":
                stop_model(model_id)
            else:
                self._json(404, {"detail": "没有这个动作"})
                return
        except RuntimeError as exc:
            self._json(400, {"detail": str(exc)})
            return
        self._json(200, snapshot())

    def _file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._json(500, {"detail": "页面文件不存在"})
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"listening on http://{HOST}:{PORT}", flush=True)
    server.serve_forever()
