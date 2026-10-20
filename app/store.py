"""转写文档存储：JSON 落盘 + 原子写 + 内存索引。"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def doc_dir(doc_id: str) -> Path:
    return config.OUTPUT_DIR / doc_id


def doc_path(doc_id: str) -> Path:
    return doc_dir(doc_id) / "transcript.json"


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class DocumentStore:
    """管理所有转写文档，线程安全。

    同一份文档可能被多个进程写入（例如命令行转写 + Web 服务），
    因此载入时会比对文件 mtime，磁盘更新则重新读取，避免内存缓存过期。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._mtimes: Dict[str, float] = {}

    # ------------------------------------------------------------ 基础读写
    def path(self, doc_id: str) -> Path:
        return doc_path(doc_id)

    def exists(self, doc_id: str) -> bool:
        return doc_path(doc_id).exists() or doc_id in self._docs

    def load(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            p = doc_path(doc_id)
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            cached = self._docs.get(doc_id)
            if cached is not None and self._mtimes.get(doc_id) == mtime:
                return cached
            if not p.exists():
                return cached
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return cached
            self._docs[doc_id] = data
            self._mtimes[doc_id] = mtime
            return data

    def save(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        doc["updated_at"] = now_iso()
        with self._lock:
            self._docs[doc["id"]] = doc
            path = doc_path(doc["id"])
            _atomic_write(path, json.dumps(doc, ensure_ascii=False, indent=2))
            try:
                self._mtimes[doc["id"]] = path.stat().st_mtime
            except OSError:
                self._mtimes.pop(doc["id"], None)
        return doc

    def update(self, doc_id: str, **fields) -> Optional[Dict[str, Any]]:
        doc = self.load(doc_id)
        if doc is None:
            return None
        doc.update(fields)
        return self.save(doc)

    def update_segment(self, doc_id: str, index: int, **fields) -> Optional[Dict[str, Any]]:
        doc = self.load(doc_id)
        if doc is None:
            return None
        segs = doc.get("segments") or []
        if index < 0 or index >= len(segs):
            return None
        segs[index].update(fields)
        return self.save(doc)

    def list_ids(self) -> List[str]:
        out = []
        if config.OUTPUT_DIR.exists():
            for d in config.OUTPUT_DIR.iterdir():
                if (d / "transcript.json").exists():
                    out.append(d.name)
        return sorted(out)

    def drop_cache(self, doc_id: str) -> None:
        with self._lock:
            self._docs.pop(doc_id, None)


def segment_final_text(seg: Dict[str, Any]) -> str:
    """取段落最终展示文本：人工编辑 > AI 润色 > 规则清理 > 原文。"""
    if seg.get("edited"):
        return seg.get("edited") or ""
    for key in ("polished", "clean", "raw", "text"):
        val = seg.get(key)
        if val:
            return val
    return ""


def new_document(
    doc_id: str,
    source_path: Path,
    media: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": doc_id,
        "source": source_path.name,
        "source_path": str(source_path),
        "media": media or {},
        "status": "pending",
        "progress": {"done": 0, "total": 0, "stage": "排队中", "message": ""},
        "asr": {},
        "segments": [],
        "knowledge_base": "",
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
