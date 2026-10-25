#!/usr/bin/env python3
"""自测：在线后端的模型列表探测（GET {base}/models）。

不起真实服务、不用 GPU：本地起一个 mock 的 OpenAI 兼容接口，
把 llm.fetch_model_list 的各条分支跑一遍。

用法：
    python scripts/test_model_probe.py
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm import _normalize_base, _parse_model_names, fetch_model_list  # noqa: E402

FAILED: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {label}: {got}")
    if not ok:
        print(f"      期望: {want}")
        FAILED.append(label)


def expect_error(label: str, fn, needle: str) -> None:
    try:
        fn()
        print(f"  ✗ {label}: 预期抛错，实际没抛")
        FAILED.append(label)
    except Exception as exc:
        hit = needle in str(exc)
        print(f"  {'✓' if hit else '✗'} {label}: {exc}")
        if not hit:
            print(f"      错误信息里应含: {needle}")
            FAILED.append(label)


class _Handler(BaseHTTPRequestHandler):
    """模拟一个 OpenAI 兼容服务。"""

    def log_message(self, *args):  # 静音
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            if self.headers.get("Authorization") != "Bearer good":
                self._send(401, {"error": "invalid api key"})
                return
            self._send(200, {"data": [{"id": "gpt-4o"}, {"id": "MiniMax-Text-01"},
                                      {"id": "gpt-4o"}]})   # 重复项应被去重
            return
        if self.path == "/models":
            self._send(200, {"models": [{"name": "ollama-model:latest"},
                                         {"model": "qwen3:8b"}]})
            return
        self._send(404, {"error": "not found"})


class _PlainHandler(BaseHTTPRequestHandler):
    """只有 /models、没有 /v1/models —— 验证自动回退。"""

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/models":
            body = json.dumps({"data": [{"id": "only-plain-models"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


def serve(handler):
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def main() -> int:
    print("· 地址归一化")
    check("补 /v1", _normalize_base("https://api.x.com"), "https://api.x.com/v1")
    check("已有 /v1 不重复补", _normalize_base("https://api.x.com/v1/"), "https://api.x.com/v1")
    check("截掉整段路径", _normalize_base("https://api.x.com/v1/chat/completions"), "https://api.x.com/v1")

    print("· 响应解析")
    check("标准 data", _parse_model_names({"data": [{"id": "a"}, {"name": "b"}]}), ["a", "b"])
    check("Ollama models", _parse_model_names({"models": [{"name": "x"}, {"model": "y"}]}), ["x", "y"])
    check("空响应", _parse_model_names({"foo": 1}), [])

    print("· 探测")
    srv, port = serve(_Handler)
    base = f"http://127.0.0.1:{port}"
    try:
        check("标准 /v1/models", fetch_model_list(base, "good"), ["gpt-4o", "MiniMax-Text-01"])
        check("地址自带 /v1", fetch_model_list(base + "/v1", "good"), ["gpt-4o", "MiniMax-Text-01"])
        expect_error("Key 无效", lambda: fetch_model_list(base, "wrong"), "鉴权失败")
        expect_error("地址为空", lambda: fetch_model_list("", ""), "Base URL")
    finally:
        srv.shutdown()

    srv2, port2 = serve(_PlainHandler)
    try:
        check("/v1/models 404 → 回退 /models", fetch_model_list(f"http://127.0.0.1:{port2}", "k"),
              ["only-plain-models"])
    finally:
        srv2.shutdown()

    print()
    if FAILED:
        print(f"✗ {len(FAILED)} 项未通过: {', '.join(FAILED)}")
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
