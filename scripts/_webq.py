#!/usr/bin/env python3
"""临时检索工具：直连 360 搜索，剥 HTML 取正文摘要。"""
from __future__ import annotations

import html
import re
import subprocess
import sys
import urllib.parse


def search(q: str, engine: str = "so360") -> str:
    if engine == "so360":
        url = "https://www.so.com/s?q=" + urllib.parse.quote(q)
    elif engine == "quark":
        url = "https://quark.sm.cn/s?q=" + urllib.parse.quote(q)
    elif engine == "toutiao":
        url = "https://so.toutiao.com/search?keyword=" + urllib.parse.quote(q)
    else:
        raise ValueError(engine)
    out = subprocess.run(
        ["curl", "-s", "-m", "25", url, "-H",
         "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"],
        capture_output=True, text=True, errors="ignore").stdout
    out = re.sub(r"(?is)<script.*?</script>|<style.*?</style>|<!--.*?-->", " ", out)
    out = re.sub(r"(?s)<[^>]+>", "\n", out)
    out = html.unescape(out)
    lines = [re.sub(r"\s+", " ", l).strip() for l in out.split("\n")]
    keep = []
    for l in lines:
        if len(l) < 12:
            continue
        if any(k in l for k in ("热搜榜", "百度搜索", "免责声明", "相关搜索", "其他人还搜")):
            continue
        keep.append(l)
    # 去重保序
    seen = set()
    uniq = []
    for l in keep:
        if l in seen:
            continue
        seen.add(l)
        uniq.append(l)
    return "\n".join(uniq)


if __name__ == "__main__":
    engine = sys.argv[1]
    for q in sys.argv[2:]:
        print(f"\n{'='*20} {engine} :: {q}")
        print(search(q, engine)[:2500])
