"""FastAPI 服务：文件列表、转写任务、段落编辑、AI 润色、音频对照、导出。"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import audio as A
from . import config
from .store import DocumentStore, segment_final_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("stt.server")

config.ensure_dirs()

app = FastAPI(title="本地语音转文字工作台", version="1.0")
store = DocumentStore()

from .pipeline import Job, Pipeline  # noqa: E402  (循环导入规避)

pipeline = Pipeline(store)

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------- 工具

def _safe_doc_id(doc_id: str) -> str:
    if not doc_id or "/" in doc_id or "\\" in doc_id or doc_id.startswith("."):
        raise HTTPException(status_code=400, detail="非法 id")
    return doc_id


def _doc_or_404(doc_id: str) -> Dict[str, Any]:
    doc = store.load(_safe_doc_id(doc_id))
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return doc


def _export_markdown(doc: Dict[str, Any], mode: str = "final") -> str:
    segs = doc.get("segments") or []
    lines = [f"# {doc.get('source', '转写稿')}", ""]
    meta = doc.get("asr") or {}
    if meta:
        lines.append(f"> 模型：{meta.get('engine')} / {meta.get('language')}　段落数：{len(segs)}")
        lines.append("")
    for seg in segs:
        ts = f"[{A.fmt_ts(seg.get('start', 0), False)}]"
        if mode == "raw":
            text = seg.get("raw", "")
        elif mode == "clean":
            text = seg.get("clean") or seg.get("raw", "")
        else:
            text = segment_final_text(seg)
        lines.append(f"{ts} {text}".rstrip())
    lines.append("")
    kb = doc.get("knowledge_base")
    if kb:
        lines += ["", "---", "", "# 知识库资料", "", kb]
    return "\n".join(lines)


def _export_plain(doc: Dict[str, Any], mode: str = "final") -> str:
    segs = doc.get("segments") or []
    out = []
    for seg in segs:
        if mode == "raw":
            text = seg.get("raw", "")
        elif mode == "clean":
            text = seg.get("clean") or seg.get("raw", "")
        else:
            text = segment_final_text(seg)
        out.append(text)
    body = "\n".join(out)
    kb = doc.get("knowledge_base")
    if kb:
        body += "\n\n" + "=" * 40 + "\n\n" + kb
    return body


def _export_srt(doc: Dict[str, Any]) -> str:
    segs = doc.get("segments") or []
    out = []
    for i, seg in enumerate(segs, 1):
        out.append(str(i))
        out.append(f"{A.fmt_ts(seg.get('start', 0), False).replace('.', ',')} --> {A.fmt_ts(seg.get('end', 0), False).replace('.', ',')}")
        out.append(segment_final_text(seg))
        out.append("")
    return "\n".join(out)


def _resolve_source(doc_id: str, name: str = "") -> Optional[Path]:
    """把文档 id 解析成 source 目录下的音频文件。

    文件列表给的是**不带扩展名**的 id（"张三-讲话"），而磁盘上是
    "张三-讲话.mp3"。以前只按 id 找，找不到就回退到 source 目录本身，
    结果探测音频直接失败——网页上点「开始转写」就是这么坏的。
    这里按「id → name → 通配符」的顺序逐个试。
    """
    direct = config.SOURCE_DIR / doc_id
    if direct.is_file():
        return direct
    if name:
        candidate = config.SOURCE_DIR / str(name)
        if candidate.is_file():
            return candidate
    # 按文件名主干匹配（支持 .mp3/.wav/.m4a 等）
    matches = sorted(p for p in config.SOURCE_DIR.glob(f"{doc_id}.*")
                     if p.is_file() and not p.name.startswith("."))
    if matches:
        return matches[0]
    return None


# ---------------------------------------------------------------- 接口

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """工作状态仪表盘（作业进度 + 硬件占用）。"""
    return HTMLResponse((STATIC_DIR / "dashboard.html").read_text(encoding="utf-8"))


@app.get("/api/status")
def status() -> Dict[str, Any]:
    """仪表盘数据：所有作业（含命令行脚本跑的）+ 系统硬件指标。

    作业状态来自 output/_status/*.json，任何进程写的都能被看到；
    系统指标用 macOS 内建命令采集，失败就返回 null（界面显示「—」）。
    """
    import os
    import time as _time
    from .monitor import snapshot
    from .status import read_all

    jobs = read_all()
    sys_info = snapshot()
    # 给每个作业挂上它自己进程的 CPU/内存
    procs = sys_info.get("processes") or {}
    for job in jobs:
        info = procs.get(str(job.get("pid")))
        if info:
            job["cpu_pct"] = info.get("cpu_pct")
            job["rss_mb"] = info.get("rss_mb")
    return {
        "server": {
            "pid": os.getpid(),
            "build": config.APP_BUILD,
            "now": _time.time(),
        },
        "jobs": jobs,
        "system": sys_info,
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "build": config.APP_BUILD,
        "source_dir": str(config.SOURCE_DIR),
        "output_dir": str(config.OUTPUT_DIR),
        "asr_model": config.ASR_MODEL,
        "llm_model": config.LLM_MODEL,
        "llm_backend": config.LLM_BACKEND,
        "language": config.ASR_LANGUAGE,
    }


@app.get("/api/llm/backends")
def llm_backends() -> Dict[str, Any]:
    """可用的润色/知识库后端。"""
    from .llm import describe_backends, load_user_config

    user = load_user_config()
    return {"default": user.get("backend") or config.LLM_BACKEND,
            "backends": describe_backends()}


@app.get("/api/llm/config")
def llm_config() -> Dict[str, Any]:
    """读取界面里保存的在线模型配置（Key 只回显末 4 位）。"""
    from .llm import load_user_config

    user = load_user_config()
    key = user.get("api_key", "")
    return {
        "backend": user.get("backend") or config.LLM_BACKEND,
        "base": user.get("base", ""),
        "model": user.get("model", ""),
        "has_key": bool(key),
        "key_hint": ("*" * 4 + key[-4:]) if key else "",
        "saved": bool(user),
        "presets": {k: {"base": v.get("base"), "model": v.get("model"),
                        "label": v.get("label"), "key_env": v.get("key_env")}
                    for k, v in config.LLM_PRESETS.items()},
    }


@app.post("/api/llm/config")
def save_llm_config(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """保存在线模型配置（存本地 .cache/llm_config.json，不进 git）。"""
    from .llm import clear_user_config, save_user_config

    if payload.get("clear"):
        clear_user_config()
        pipeline.reset_llm_cache()
        return {"ok": True, "cleared": True, "message": "已清除在线模型配置"}
    saved = save_user_config(payload)
    pipeline.reset_llm_cache()
    return {"ok": True, "saved": True,
            "backend": saved.get("backend", ""),
            "has_key": bool(saved.get("api_key")),
            "message": "已保存，后续润色将使用该后端"}


@app.post("/api/llm/models")
def llm_models(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """探测在线服务可用的模型列表（填好地址/Key 就能拉，不用背模型名）。

    传上来的 base/api_key 优先（未保存也能试），
    没传则回落到已保存配置或该后端的预设地址。
    """
    from .llm import fetch_model_list, resolve_probe_target

    backend = str(payload.get("backend") or "").strip().lower() or "local"
    override = {k: str(payload.get(k) or "").strip() for k in ("base", "api_key")}
    override = {k: v for k, v in override.items() if v}
    base, api_key = resolve_probe_target(backend, override)
    try:
        names = fetch_model_list(base, api_key=api_key)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "base": base, "count": len(names), "models": names}


@app.post("/api/llm/test")
def test_llm(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """连通性自检。可带上界面里尚未保存的 base/model/api_key 直接试连。"""
    from .llm import test_connection

    backend = str(payload.get("backend") or "").strip() or None
    override = {k: str(payload.get(k) or "").strip() for k in ("base", "model", "api_key")}
    override = {k: v for k, v in override.items() if v}
    if backend in (None, "", "local", "mlx", "mlx-lm"):
        # 本地后端：复用管线里已缓存的实例（别再加载一份 8B 权重，16GB 机器会 OOM）
        return test_connection("local", llm=pipeline.get_llm("local"))
    return test_connection(backend, override=override or None)


@app.get("/api/files")
def list_files() -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    if config.SOURCE_DIR.exists():
        for p in sorted(config.SOURCE_DIR.iterdir()):
            if not p.is_file() or p.suffix.lower() not in config.AUDIO_EXTS:
                continue
            doc_id = p.stem
            doc = store.load(doc_id)
            info = {}
            if doc:
                info = {
                    "status": doc.get("status"),
                    "progress": doc.get("progress"),
                    "segments": len(doc.get("segments") or []),
                    "has_kb": bool(doc.get("knowledge_base")),
                }
            items.append({
                "id": doc_id,
                "name": p.name,
                "size": p.stat().st_size,
                "ext": p.suffix.lower(),
                "doc": info,
            })
    return {"source_dir": str(config.SOURCE_DIR), "files": items}


@app.get("/api/docs")
def list_docs() -> Dict[str, Any]:
    out = []
    for doc_id in store.list_ids():
        doc = store.load(doc_id)
        if not doc:
            continue
        out.append({
            "id": doc_id,
            "source": doc.get("source"),
            "status": doc.get("status"),
            "segments": len(doc.get("segments") or []),
            "updated_at": doc.get("updated_at"),
        })
    return {"docs": out}


@app.get("/api/doc/{doc_id}")
def get_doc(doc_id: str) -> Dict[str, Any]:
    doc = _doc_or_404(doc_id)
    job = pipeline.get_job(doc_id)
    doc = dict(doc)
    doc["job"] = {
        "stage": job.stage,
        "message": job.message,
        "running": job.running,
        "error": job.error,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    } if job else None
    return doc


@app.post("/api/transcribe")
def start_transcribe(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    doc_id = str(payload.get("id") or "").strip()
    if not doc_id:
        raise HTTPException(status_code=400, detail="缺少文件 id")
    source = _resolve_source(doc_id, payload.get("name"))
    if source is None:
        raise HTTPException(status_code=404, detail=f"找不到源文件: {doc_id}")

    engine = str(payload.get("engine") or "qwen3")
    language = payload.get("language") or None
    align = bool(payload.get("align", True))
    use_rules = bool(payload.get("rules", True))
    force = bool(payload.get("force", False))
    hotwords = [str(h) for h in (payload.get("hotwords") or []) if str(h).strip()]

    job = pipeline.start_transcribe(
        doc_id=doc_id, source_path=source, engine=engine, language=language,
        align=align, use_rules=use_rules, force=force, hotwords=hotwords,
    )
    return {"id": doc_id, "stage": job.stage, "message": job.message}


@app.post("/api/doc/{doc_id}/segment/{index}")
def edit_segment(doc_id: str, index: int, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """保存人工修改。field: raw = 改原文；edited = 改润色稿（默认）。"""
    text = payload.get("text")
    if text is None:
        raise HTTPException(status_code=400, detail="缺少 text")
    field = str(payload.get("field") or "edited")
    if field not in ("raw", "edited"):
        raise HTTPException(status_code=400, detail="不支持的字段")
    doc = store.load(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    segs = doc.get("segments") or []
    if index < 0 or index >= len(segs):
        raise HTTPException(status_code=404, detail="段落不存在")
    seg = segs[index]
    if field == "raw":
        seg["raw"] = str(text)
        seg["raw_edited"] = True
        # 原文改动后，规则清理稿同步重算
        from .textproc import clean_text

        seg["clean"] = clean_text(str(text), level="standard")
    else:
        seg["edited"] = str(text)
        seg["edited_manually"] = True
    store.save(doc)
    return {"ok": True, "segment": seg}


@app.post("/api/doc/{doc_id}/segment/{index}/reset")
def reset_segment(doc_id: str, index: int) -> Dict[str, Any]:
    doc = _doc_or_404(doc_id)
    segs = doc.get("segments") or []
    if index < 0 or index >= len(segs):
        raise HTTPException(status_code=404, detail="段落不存在")
    segs[index].pop("edited", None)
    segs[index].pop("edited_manually", None)
    store.save(doc)
    return {"ok": True, "segment": segs[index]}


@app.post("/api/doc/{doc_id}/rules")
def run_rules(doc_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    level = str(payload.get("level") or "light")
    doc = pipeline.run_rules(doc_id, level=level)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"ok": True, "segments": len(doc.get("segments") or [])}


@app.post("/api/doc/{doc_id}/polish")
def run_polish(doc_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """后台启动润色任务，立即返回；进度通过 /api/doc/{doc_id} 查看。"""
    indices = payload.get("indices")
    idxs = [int(i) for i in indices] if isinstance(indices, list) else None
    level = str(payload.get("level") or "standard")
    force = bool(payload.get("force", False))
    backend = str(payload.get("backend") or "").strip() or None

    with pipeline._lock:
        running = pipeline.jobs.get(doc_id)
        if running and running.running:
            return {"ok": True, "started": False, "message": "该文档已有任务在运行"}

    def runner():
        job = Job(doc_id=doc_id)
        job.stage = "polishing"
        job.message = "本地 LLM 润色中" if not backend else f"{backend} 润色中"
        job.started_at = time.time()
        pipeline.jobs[doc_id] = job
        try:
            pipeline.run_polish(doc_id, indices=idxs, level=level,
                                force_polish=force, backend=backend)
            job.stage = "done"
            job.message = "润色完成"
        except Exception as exc:
            log.exception("润色失败")
            job.stage = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = job.error
        finally:
            job.finished_at = time.time()

    th = threading.Thread(target=runner, name=f"polish-{doc_id}", daemon=True)
    with pipeline._lock:
        pipeline._threads[doc_id] = th
    th.start()
    return {"ok": True, "started": True, "message": "润色任务已启动"}


@app.post("/api/doc/{doc_id}/knowledge_base")
def run_kb(doc_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """后台启动知识库提炼，立即返回。"""
    title = str(payload.get("title") or "")
    backend = str(payload.get("backend") or "").strip() or None
    doc = _doc_or_404(doc_id)

    def runner():
        job = Job(doc_id=doc_id)
        job.stage = "knowledge_base"
        job.message = "提炼知识库中"
        job.started_at = time.time()
        pipeline.jobs[doc_id] = job
        try:
            pipeline.run_knowledge_base(doc_id, title=title, backend=backend)
            job.stage = "done"
            job.message = "知识库完成"
        except Exception as exc:
            log.exception("知识库生成失败")
            job.stage = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = job.error
        finally:
            job.finished_at = time.time()

    th = threading.Thread(target=runner, name=f"kb-{doc_id}", daemon=True)
    with pipeline._lock:
        pipeline._threads[doc_id] = th
    th.start()
    return {"ok": True, "started": True, "message": "知识库任务已启动"}


@app.get("/api/doc/{doc_id}/export")
def export_doc(doc_id: str, fmt: str = Query("md"), mode: str = Query("final")):
    doc = _doc_or_404(doc_id)
    base = doc_id
    if fmt == "md":
        body = _export_markdown(doc, mode)
        media = "text/markdown; charset=utf-8"
        name = f"{base}.md"
    elif fmt == "txt":
        body = _export_plain(doc, mode)
        media = "text/plain; charset=utf-8"
        name = f"{base}.txt"
    elif fmt == "srt":
        body = _export_srt(doc)
        media = "application/x-subrip; charset=utf-8"
        name = f"{base}.srt"
    else:
        raise HTTPException(status_code=400, detail="不支持的格式")
    # 中文文件名需要 RFC 5987 编码，同时给出 ASCII 回退名
    quoted = urllib.parse.quote(name)
    fallback = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).stem) or "transcript"
    disposition = f"attachment; filename=\"{fallback}{Path(name).suffix}\"; filename*=UTF-8''{quoted}"
    return Response(
        content=body.encode("utf-8"),
        media_type=media,
        headers={"Content-Disposition": disposition},
    )


@app.get("/api/doc/{doc_id}/audio")
def stream_audio(doc_id: str):
    _safe_doc_id(doc_id)
    path = config.OUTPUT_DIR / doc_id / "preview.mp3"
    if not path.exists():
        path = _find_source(doc_id)
        if path is None:
            raise HTTPException(status_code=404, detail="音频不存在")
    return _range_file(path)


def _find_source(doc_id: str) -> Optional[Path]:
    """按 stem 找回原始音频（预览文件尚未生成时兜底）。"""
    direct = config.SOURCE_DIR / doc_id
    if direct.exists():
        return direct
    for ext in config.AUDIO_EXTS:
        cand = config.SOURCE_DIR / f"{doc_id}{ext}"
        if cand.exists():
            return cand
    return None


@app.get("/api/doc/{doc_id}/peaks")
def get_peaks(doc_id: str):
    _safe_doc_id(doc_id)
    path = config.OUTPUT_DIR / doc_id / "peaks.bin"
    if not path.exists():
        raise HTTPException(status_code=404, detail="波形数据不存在")
    return Response(content=path.read_bytes(), media_type="application/octet-stream")


@app.get("/api/doc/{doc_id}/original")
def stream_original(doc_id: str):
    _safe_doc_id(doc_id)
    path = config.SOURCE_DIR / doc_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="原始音频不存在")
    return _range_file(path)


def _range_file(path: Path) -> Response:
    """支持 HTTP Range 的文件响应（浏览器 seek 必需）。"""
    size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return _StreamingRange(path, size, media_type)


from starlette.requests import Request  # noqa: E402


class _StreamingRange(Response):
    """按请求 Range 头返回 206/200。"""

    def __init__(self, path: Path, size: int, media_type: str):
        super().__init__(media_type=media_type)
        self.path = path
        self.size = size

    async def __call__(self, scope, receive, send):  # type: ignore[override]
        request = Request(scope, receive)
        range_header = request.headers.get("range")
        start, end = 0, self.size - 1
        status = 200
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), self.size - 1)
                elif m.group(2):
                    start = max(0, self.size - int(m.group(2)))
                if start >= self.size:
                    start, end = 0, self.size - 1
                status = 206
        length = end - start + 1

        async def iterator():
            with open(self.path, "rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    data = fh.read(min(1024 * 512, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Length": str(length),
            "Accept-Ranges": "bytes",
            "Content-Type": self.media_type or "application/octet-stream",
        }
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{self.size}"
        await send({"type": "http.response.start", "status": status, "headers": [
            (k.encode(), str(v).encode()) for k, v in headers.items()
        ]})
        async for chunk in iterator():
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")


if __name__ == "__main__":
    run()
