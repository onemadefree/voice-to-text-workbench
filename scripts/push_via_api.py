#!/usr/bin/env python3
"""在 github.com 不可直连时，通过 Git Database API 完成推送。

背景：部分网络环境下 api.github.com 可达而 github.com:443 被阻断，
git push 无法完成。本脚本改走 api.github.com：
引导提交 → 上传 blob → 建 tree → 建 commit → 更新 ref。

为保证本地与远端 SHA 一致，本地按远端返回的元数据**逐字节重建**提交对象
（tree / parent / author / committer / encoding / message），再以
`git hash-object -t commit -w` 写入本地对象库；SHA 由内容寻址天然一致。

用法：
    python scripts/push_via_api.py [owner/repo] [--branch=main]
"""

from __future__ import annotations

import base64
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REPO = "onemadefree/voice-to-text-workbench"
BRANCH = "main"
TOKEN = ""


# ------------------------------------------------------------------ 基础工具

def gh_token() -> str:
    r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    tok = r.stdout.strip()
    if not tok:
        raise SystemExit("取不到 gh 凭据，请先运行 gh auth login")
    return tok


def api(method: str, path: str, payload=None):
    url = f"https://api.github.com/repos/{REPO}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "push-via-api")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:400]
        raise SystemExit(f"{method} {path} → HTTP {exc.code}\n{detail}")


def git(*args: str, input_text: str | None = None) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                       input=input_text, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失败:\n{r.stderr.strip()[:400]}")
    return r.stdout.strip()


def local_entries() -> list[tuple[str, str]]:
    """已跟踪文件：(mode, path)"""
    out = git("ls-files", "-s", "-z")
    entries = []
    for rec in out.split("\0"):
        if not rec:
            continue
        meta, path = rec.split("\t", 1)
        mode, _sha, _stage = meta.split(" ", 2)
        entries.append((mode, path))
    return entries


def build_tree(entries: list[tuple[str, str, str]]) -> str:
    """用临时 index 建树，支持嵌套路径（git mktree 不接受带斜杠的路径）。"""
    with tempfile.TemporaryDirectory() as tmp:
        index = os.path.join(tmp, "index")
        env = {**os.environ, "GIT_INDEX_FILE": index}
        subprocess.run(["git", "read-tree", "--empty"], cwd=ROOT, env=env,
                       capture_output=True, check=True)
        for mode, sha, path in entries:
            subprocess.run(
                ["git", "update-index", "--add", "--cacheinfo", f"{mode},{sha},{path}"],
                cwd=ROOT, env=env, capture_output=True, check=True,
            )
        out = subprocess.run(["git", "write-tree"], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()


def write_commit_object(raw: bytes) -> str:
    """把原始 commit 字节写入本地对象库，返回其 SHA。"""
    r = subprocess.run(["git", "hash-object", "-t", "commit", "-w", "--stdin"],
                       cwd=ROOT, input=raw, capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"写入 commit 对象失败: {r.stderr.decode('utf-8', 'ignore')[:300]}")
    return r.stdout.decode().strip()


# ------------------------------------------------------------------ 时间处理

_ISO_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(\.\d+)?([+-])(\d{2}):?(\d{2})$")


def parse_iso_date(value: str) -> tuple[int, str]:
    """'2026-10-21T12:35:41+08:00' / '...Z' → (epoch秒, '+0800')"""
    value = value.strip()
    if value.endswith("Z"):
        base = value[:-1]
        m = re.match(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})", base)
        if not m:
            raise ValueError(f"无法解析日期: {value}")
        dt = datetime.datetime.strptime(f"{m.group(1)} {m.group(2)}",
                                        "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=datetime.timezone.utc)
        return int(dt.timestamp()), "+0000"
    m = _ISO_RE.match(value)
    if not m:
        raise ValueError(f"无法解析日期: {value}")
    sign = 1 if m.group(4) == "+" else -1
    tz = datetime.timezone(sign * datetime.timedelta(hours=int(m.group(5)),
                                                     minutes=int(m.group(6))))
    dt = datetime.datetime.strptime(f"{m.group(1)} {m.group(2)}",
                                    "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
    return int(dt.timestamp()), f"{m.group(4)}{m.group(5)}{m.group(6)}"


def local_offsets() -> list[str]:
    """本机时区偏移（如 '+0800'）。GitHub 存提交时保留作者时区，
    但 API 返回的日期串会统一标成 Z，必须把本地偏移也列入候选。"""
    out = []
    tz = datetime.datetime.now().astimezone().strftime("%z")
    if tz and tz != "+0000":
        out.append(tz)
    return out


def ident_to_iso(value: str) -> dict:
    """'Name <email> 1750000000 +0800' → API 用的 author/committer 字典"""
    name, _, rest = value.partition(" <")
    email, _, stamp = rest.partition("> ")
    ts, _, offset = stamp.partition(" ")
    offset = offset.strip() or "+0000"
    sign = 1 if offset.startswith("+") else -1
    total = int(ts) - sign * (int(offset[1:3]) * 3600 + int(offset[3:5]) * 60)
    dt = datetime.datetime.fromtimestamp(total, datetime.timezone.utc)
    iso = dt.strftime("%Y-%m-%dT%H:%M:%S") + f"{offset[:3]}:{offset[3:]}"
    return {"name": name, "email": email, "date": iso, "ts": int(ts), "offset": offset}


def raw_commit_bytes(tree: str, parents: list[str], message: bytes,
                     author: tuple[str, str, int, str],
                     committer: tuple[str, str, int, str],
                     encoding: bool) -> bytes:
    head = [f"tree {tree}"]
    head += [f"parent {p}" for p in parents]
    head.append(f"author {author[0]} <{author[1]}> {author[2]} {author[3]}")
    head.append(f"committer {committer[0]} <{committer[1]}> {committer[2]} {committer[3]}")
    if encoding:
        head.append("encoding UTF-8")
    return ("\n".join(head) + "\n\n").encode("utf-8") + message


# ------------------------------------------------------------------ 提交复刻

def reconstruct_commit(sha: str) -> str:
    """按远端元数据逐字节重建提交对象；返回本地 SHA（与远端一致）。"""
    meta = api("GET", f"/git/commits/{sha}")
    tree_obj = meta["tree"]
    tree = tree_obj["sha"] if isinstance(tree_obj, dict) else tree_obj
    parents = [p["sha"] for p in meta.get("parents", [])]
    message = meta.get("message", "")

    a_ts, a_off = parse_iso_date(meta["author"]["date"])
    c_ts, c_off = parse_iso_date(meta["committer"]["date"])
    a_offs = list(dict.fromkeys([a_off, "+0000"] + local_offsets()))
    c_offs = list(dict.fromkeys([c_off, "+0000"] + local_offsets()))
    msg_variants = list(dict.fromkeys([message, message + "\n",
                                       message.rstrip("\n") + "\n"]))

    for encoding in (False, True):
        for msg in msg_variants:
            for ao in a_offs:
                for co in c_offs:
                    raw = raw_commit_bytes(
                        tree, parents, msg.encode("utf-8"),
                        (meta["author"]["name"], meta["author"]["email"], a_ts, ao),
                        (meta["committer"]["name"], meta["committer"]["email"], c_ts, co),
                        encoding)
                    local = write_commit_object(raw)
                    if local == sha:
                        return local
    raise SystemExit(f"无法在本地复刻提交 {sha}")


def remote_head() -> str | None:
    try:
        ref = api("GET", f"/git/ref/heads/{BRANCH}")
        return ref["object"]["sha"]
    except SystemExit:
        return None


def bootstrap() -> str:
    """空仓库时先用 Contents API 建一个引导提交。"""
    print("· 仓库为空，先创建引导提交", flush=True)
    content = (ROOT / ".gitignore").read_bytes()
    api("PUT", "/contents/.gitignore", {
        "message": "chore: 初始化仓库",
        "content": base64.b64encode(content).decode("ascii"),
        "branch": BRANCH,
    })
    head = remote_head()
    if not head:
        raise SystemExit("引导提交后仍取不到分支，请检查仓库状态")
    return head


# ------------------------------------------------------------------ 主流程

def main() -> int:
    global REPO, BRANCH, TOKEN
    argv = sys.argv[1:]
    for a in argv:
        if a.startswith("--branch="):
            BRANCH = a.split("=", 1)[1]
    plain = [a for a in argv if not a.startswith("--")]
    if plain:
        REPO = plain[0]
    TOKEN = gh_token()

    entries = local_entries()
    print(f"仓库 {REPO}　分支 {BRANCH}　文件 {len(entries)} 个", flush=True)

    base = remote_head()
    if not base:
        base = bootstrap()

    # 1. 在本地复刻远端当前 head（新提交的父）
    base_local = reconstruct_commit(base)
    print(f"  复刻基线 {base[:10]} → {base_local[:10]} "
          f"{'✓ 一致' if base_local == base else '✗ 不一致'}", flush=True)

    # 2. 上传 blob
    tree_entries = []
    for mode, path in entries:
        data = (ROOT / path).read_bytes()
        blob = api("POST", "/git/blobs", {
            "content": base64.b64encode(data).decode("ascii"),
            "encoding": "base64",
        })
        tree_entries.append({"path": path, "mode": mode, "type": "blob", "sha": blob["sha"]})
    print(f"  上传 blob {len(tree_entries)} 个", flush=True)

    # 3. 建 tree（与本地 tree 内容一致，SHA 应相同）
    tree = api("POST", "/git/trees", {"tree": tree_entries})
    local_tree = build_tree([(e["mode"], e["sha"], e["path"]) for e in tree_entries])
    print(f"  tree {tree['sha'][:10]}　本地重建 {local_tree[:10]} "
          f"{'✓ 一致' if local_tree == tree['sha'] else '✗ 不一致'}", flush=True)
    if local_tree != tree["sha"]:
        raise SystemExit("远端 tree 与本地重建 tree 不一致，终止")

    # 4. 建 commit（父 = 远端当前 head）
    head = git("cat-file", "commit", "HEAD")
    head_meta, _, message = head.partition("\n\n")
    author = committer = None
    for line in head_meta.splitlines():
        key, _, val = line.partition(" ")
        if key == "author":
            author = ident_to_iso(val)
        elif key == "committer":
            committer = ident_to_iso(val)
    payload_author = {k: author[k] for k in ("name", "email", "date")}
    payload_committer = {k: committer[k] for k in ("name", "email", "date")}
    commit = api("POST", "/git/commits", {
        "message": message,
        "tree": tree["sha"],
        "parents": [base],
        "author": payload_author,
        "committer": payload_committer,
    })
    print(f"  commit {commit['sha'][:10]}", flush=True)

    # 5. 更新 ref
    api("PATCH", f"/git/refs/heads/{BRANCH}", {"sha": commit["sha"], "force": True})
    print(f"  ref refs/heads/{BRANCH} → {commit['sha'][:10]}", flush=True)

    # 6. 本地复刻新提交并对齐分支
    local_commit = reconstruct_commit(commit["sha"])
    same = local_commit == commit["sha"]
    print(f"  本地 {local_commit[:10]} {'✓ 与远端一致' if same else '✗ 不一致'}")
    if not same:
        raise SystemExit("本地与远端 SHA 不一致，终止")

    git("update-ref", f"refs/heads/{BRANCH}", local_commit)
    git("update-ref", f"refs/remotes/origin/{BRANCH}", commit["sha"])
    subprocess.run(["git", "reset", "--mixed", local_commit], cwd=ROOT, check=False,
                   capture_output=True)
    print(f"\n✓ 推送完成：https://github.com/{REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
