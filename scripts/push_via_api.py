#!/usr/bin/env python3
"""在 github.com 不可直连时，通过 Git Database API 完成推送。

背景：部分网络环境下 api.github.com 可达而 github.com:443 被阻断，
git push 无法完成。本脚本改走 api.github.com：
引导提交 → 上传 blob → 建 tree → 建 commit → 更新 ref。

为保证本地与远端 SHA 一致，本地按远端返回的元数据**逐字节重建**提交对象
（tree / parent / author / committer / encoding / message），再以
`git hash-object -t commit -w` 写入本地对象库；SHA 由内容寻址天然一致。

用法：
    python scripts/push_via_api.py [owner/repo] [--branch=main] [--dry-run]
    python scripts/push_via_api.py --help

⚠️ 本脚本会**直接改写远端分支引用**。运行前请确认：
    1. `--branch` 指向的远端分支确实是你要覆盖的分支；
    2. 想先看结果就加 `--dry-run`（只做检查和上传，不改远端引用）。
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
DRY_RUN = False
USE_WORKTREE = False


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


def git_bytes(*args: str) -> bytes:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失败:\n{r.stderr.decode('utf-8', 'ignore')[:400]}")
    return r.stdout


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


def ref_entries(ref: str) -> list[tuple[str, str, str]]:
    """指定 ref 的已提交内容：(mode, path, blob_sha)。

    用它而不是工作区，是为了保证「推的是这个分支已提交的内容」——
    否则在别的分支上跑本脚本会把当前工作区的文件推到目标分支上。
    """
    out = git_bytes("ls-tree", "-r", "-z", ref)
    entries: list[tuple[str, str, str]] = []
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, kind, sha = meta.decode().split(" ", 2)
        if kind != "blob":
            continue  # 只推 blob（文件），子模块等类型跳过
        entries.append((mode, path.decode("utf-8"), sha))
    return entries


def build_tree(entries: list[tuple[str, str, str]]) -> str:
    """用临时 index 建树，支持嵌套路径（git mktree 不接受带斜杠的路径）。"""
    # 临时 index 放在仓库内：沙箱里 /tmp 可能不可写，git read-tree 会失败
    with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
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
    """'Name <email> 1750000000 +0800' → API 用的 author/committer 字典

    时间戳与偏移必须自洽：ISO 串里写的是「墙钟 + 偏移」，
    墙钟要用**作者所在时区**的，不能拿 UTC 墙钟贴本地偏移——
    那样 GitHub 会按字面时刻收下，推上去的提交时间整体差一个时区。
    """
    name, _, rest = value.partition(" <")
    email, _, stamp = rest.partition("> ")
    ts, _, offset = stamp.partition(" ")
    offset = offset.strip() or "+0000"
    sign = 1 if offset.startswith("+") else -1
    off_delta = datetime.timedelta(hours=int(offset[1:3]), minutes=int(offset[3:5]))
    epoch = int(ts)
    # ts 本身就是绝对时刻；作者所在时区的墙钟 = epoch + 偏移
    local_dt = datetime.datetime.fromtimestamp(
        epoch + sign * int(off_delta.total_seconds()), datetime.timezone.utc)
    iso = local_dt.strftime("%Y-%m-%dT%H:%M:%S") + f"{offset[:3]}:{offset[3:]}"
    return {"name": name, "email": email, "date": iso, "ts": epoch, "offset": offset}


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

def parse_args(argv: list[str]) -> None:
    """严格解析参数：未知参数直接报错，绝不静默忽略。

    （旧版本会忽略一切 `--xxx`，导致 `--help` 也照样执行一次真实推送。）
    """
    global REPO, BRANCH, TOKEN, DRY_RUN, USE_WORKTREE
    plain: list[str] = []
    for a in argv:
        if a in ("-h", "--help"):
            print(__doc__)
            raise SystemExit(0)
        if a == "--dry-run":
            DRY_RUN = True
            continue
        if a == "--worktree":
            USE_WORKTREE = True
            continue
        if a.startswith("--branch="):
            BRANCH = a.split("=", 1)[1]
            continue
        if a.startswith("-"):
            raise SystemExit(f"未知参数 {a!r}；用法见 --help")
        plain.append(a)
    if len(plain) > 1:
        raise SystemExit(f"最多接受一个 owner/repo 参数，收到 {len(plain)} 个")
    if plain:
        REPO = plain[0]


def pick_source() -> tuple[str, str]:
    """决定推哪个本地提交，返回 (提交 SHA, 来源说明)。

    默认推 **目标分支已提交的内容**——这样在别的分支上跑本脚本，
    不会把当前工作区的文件错推到目标分支上（这是旧版本最危险的行为）。
    """
    source_ref = f"refs/heads/{BRANCH}"
    if USE_WORKTREE:
        cur = git("rev-parse", "--abbrev-ref", "HEAD")
        if cur != BRANCH:
            raise SystemExit(
                f"--worktree 要求当前就在 {BRANCH} 上（当前在 {cur}）。\n"
                "不指定 --worktree 时推的是该分支已提交的内容，更安全。")
        return git("rev-parse", "HEAD"), "工作区内容（含未提交改动）"

    if git("rev-parse", "--verify", "--quiet", source_ref):
        return git("rev-parse", source_ref), f"{source_ref} 已提交内容"
    # 本地没有这个分支 → 退到 HEAD，并在输出里明说
    return git("rev-parse", "HEAD"), "HEAD 已提交内容（本地无该分支）"


def git_ok(*args: str) -> bool:
    """跑一条 git 命令，只关心成败（不抛异常）。"""
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    return r.returncode == 0


def plan_chain(base: str, tip: str) -> list[str]:
    """列出需要依次在远端创建的本地提交（不含 base）。

    base 是远端 head、tip 是本地要推的提交。
    base 是 tip 的祖先时 → 返回真实的提交链（保留每个提交各自的说明）；
    否则（分叉 / 强推）→ 返回空，走单提交压平路径。
    """
    if not git_ok("merge-base", "--is-ancestor", base, tip):
        return []
    out = git("rev-list", "--reverse", f"{base}..{tip}")
    return [c for c in out.split("\n") if c]


def commit_meta(sha: str) -> tuple[str, dict, dict]:
    """取一个提交的 (说明, author, committer)，供 API 建提交用。"""
    raw = git("cat-file", "commit", sha)
    head_meta, _, message = raw.partition("\n\n")
    author = committer = None
    for line in head_meta.splitlines():
        key, _, val = line.partition(" ")
        if key == "author":
            author = ident_to_iso(val)
        elif key == "committer":
            committer = ident_to_iso(val)
    if not author or not committer:
        raise SystemExit(f"提交 {sha[:10]} 的 author/committer 解析失败")
    keys = ("name", "email", "date")
    return message, {k: author[k] for k in keys}, {k: committer[k] for k in keys}


def upload_blobs(entries_list: list[list[tuple[str, str, str]]]) -> None:
    """上传所有提交链用到的 blob（按 SHA 去重，只传一次）。"""
    todo: dict[str, str] = {}          # blob sha → 路径（用于报错定位）
    for entries in entries_list:
        for _mode, path, sha in entries:
            todo.setdefault(sha, path)
    for sha, path in todo.items():
        data = git_bytes("cat-file", "blob", sha)
        blob = api("POST", "/git/blobs", {
            "content": base64.b64encode(data).decode("ascii"),
            "encoding": "base64",
        })
        # blob 字节直接取自本地对象库，内容寻址保证 SHA 必然一致
        if sha != blob["sha"]:
            raise SystemExit(f"{path}: 本地 blob {sha[:10]} 与远端 {blob['sha'][:10]} 不一致")
    print(f"  上传 blob {len(todo)} 个", flush=True)


def create_commit(local_sha: str, parent: str) -> str:
    """在远端按本地提交的原样内容建一个提交，返回远端 SHA。

    ⚠️ GitHub 的建提交接口会**剥掉说明末尾的换行**，而 git 本地提交是带换行的，
    所以远端 SHA 与本地 SHA 必然不同（差一个末尾 \\n）。这是接口行为，
    不是错误——脚本后面会按远端返回的字节在本地重建一份并对齐分支。
    """
    entries = ref_entries(local_sha)
    tree = api("POST", "/git/trees", {
        "tree": [{"path": p, "mode": m, "type": "blob", "sha": s} for m, p, s in entries]
    })
    local_tree = git("rev-parse", f"{local_sha}^{{tree}}")
    if tree["sha"] != local_tree:
        raise SystemExit(f"提交 {local_sha[:10]} 的 tree 不一致："
                         f"远端 {tree['sha'][:10]} vs 本地 {local_tree[:10]}")
    message, author, committer = commit_meta(local_sha)
    commit = api("POST", "/git/commits", {
        "message": message,
        "tree": tree["sha"],
        "parents": [parent],
        "author": author,
        "committer": committer,
    })
    print(f"  commit {local_sha[:10]} → 远端 {commit['sha'][:10]}", flush=True)
    return commit["sha"]


def main() -> int:
    global TOKEN
    parse_args(sys.argv[1:])
    TOKEN = gh_token()

    tip, source_desc = pick_source()
    print(f"仓库 {REPO}　分支 {BRANCH}　来源 {source_desc}", flush=True)

    base = remote_head()
    if not base:
        base = bootstrap()

    # 1. 在本地复刻远端当前 head（新提交的父）
    base_local = reconstruct_commit(base)
    print(f"  复刻基线 {base[:10]} → {base_local[:10]} "
          f"{'✓ 一致' if base_local == base else '✗ 不一致'}", flush=True)

    # 2. 规划要推送的提交链
    chain = [] if USE_WORKTREE else plan_chain(base, tip)
    if chain:
        print(f"  待推送提交链 {len(chain)} 个（保留每个提交各自的说明）", flush=True)
        upload_blobs([ref_entries(c) for c in chain])
        final = base
        for c in chain:
            final = create_commit(c, final)
    else:
        # 单提交路径：工作区内容，或远端与本地已分叉（需要压平）
        print("  单提交路径（压平为 1 个提交）", flush=True)
        entries = [(m, p, git("hash-object", "-w", "--", p)) for m, p in local_entries()]
        upload_blobs([entries])
        tree = api("POST", "/git/trees", {
            "tree": [{"path": p, "mode": m, "type": "blob", "sha": s} for m, p, s in entries]
        })
        local_tree = build_tree([(e[0], e[2], e[1]) for e in entries])
        print(f"  tree {tree['sha'][:10]}　本地重建 {local_tree[:10]} "
              f"{'✓ 一致' if local_tree == tree['sha'] else '✗ 不一致'}", flush=True)
        if local_tree != tree["sha"]:
            raise SystemExit("远端 tree 与本地重建 tree 不一致，终止")
        message, author, committer = commit_meta(tip)
        commit = api("POST", "/git/commits", {
            "message": message,
            "tree": tree["sha"],
            "parents": [base],
            "author": author,
            "committer": committer,
        })
        print(f"  commit {commit['sha'][:10]}", flush=True)
        final = commit["sha"]

    if DRY_RUN:
        print(f"\n[dry-run] 未改远端引用。将会把 refs/heads/{BRANCH} → {final[:10]}")
        print("[dry-run] 去掉 --dry-run 再跑一次即可真正推送。")
        return 0

    # 3. 更新 ref
    api("PATCH", f"/git/refs/heads/{BRANCH}", {"sha": final, "force": True})
    print(f"  ref refs/heads/{BRANCH} → {final[:10]}", flush=True)

    # 4. 本地复刻新提交并对齐分支
    local_commit = reconstruct_commit(final)
    same = local_commit == final
    print(f"  本地 {local_commit[:10]} {'✓ 与远端一致' if same else '✗ 不一致'}")
    if not same:
        raise SystemExit("本地与远端 SHA 不一致，终止")

    git("update-ref", f"refs/heads/{BRANCH}", local_commit)
    git("update-ref", f"refs/remotes/origin/{BRANCH}", final)
    # 只有当 HEAD 本来就在这个分支上时才同步工作区；
    # 否则绝不能 reset——那会把当前分支拽到别的分支的提交上。
    cur = git("rev-parse", "--abbrev-ref", "HEAD")
    if cur == BRANCH:
        subprocess.run(["git", "reset", "--mixed", local_commit], cwd=ROOT, check=False,
                       capture_output=True)
    print(f"\n✓ 推送完成：https://github.com/{REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
