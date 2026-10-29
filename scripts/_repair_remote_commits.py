"""补齐本地对象库里缺失的远端提交与 tree 对象。

背景：push_via_api.py 走 API 推送时，远端提交的 commit 对象是在本地重建的，
但**引用的 tree 对象不一定被写进本地库**（仓库初始化时的根提交就是这种情况：
`chore: 初始化仓库` 的 tree 只存在于 GitHub）。
后果：`git show <根提交>` / `git diff` 碰到它会报 "not a tree object"，
`git fsck` 报 broken link。

用法：
    python scripts/_repair_remote_commits.py <sha> [<sha> ...]
    python scripts/_repair_remote_commits.py --chain <tip-sha>
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pva", ROOT / "scripts" / "push_via_api.py")
pva = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pva)
pva.TOKEN = pva.gh_token()


def chain(tip: str) -> list[str]:
    """从远端 tip 一路回溯到根，返回由旧到新的提交列表。"""
    out = []
    cur = tip
    while cur:
        meta = pva.api("GET", f"/git/commits/{cur}")
        parents = [p["sha"] for p in meta.get("parents", [])]
        out.append(cur)
        cur = parents[0] if parents else None
    return list(reversed(out))


def ensure_tree(tree_sha: str) -> bool:
    """确保 tree 对象在本地存在；缺失就从 GitHub 拉下来重建。"""
    if subprocess.run(["git", "cat-file", "-e", f"{tree_sha}^{{tree}}"],
                      cwd=ROOT, capture_output=True).returncode == 0:
        return False
    meta = pva.api("GET", f"/git/trees/{tree_sha}")
    lines = []
    for e in meta.get("tree", []):
        if e["type"] != "blob":
            continue
        # blob 也要在本地
        if subprocess.run(["git", "cat-file", "-e", e["sha"]],
                          cwd=ROOT, capture_output=True).returncode != 0:
            data = pva.api("GET", f"/git/blobs/{e['sha']}")
            import base64
            raw = base64.b64decode(data["content"])
            r = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=ROOT,
                               input=raw, capture_output=True)
            if r.stdout.decode().strip() != e["sha"]:
                raise SystemExit(f"blob {e['sha'][:10]} 重建后 SHA 不一致")
        lines.append(f"{e['mode']} blob {e['sha']}\t{e['path']}")
    if not lines:
        raise SystemExit(f"tree {tree_sha[:10]} 为空，无法重建")
    r = subprocess.run(["git", "mktree"], cwd=ROOT,
                       input=("\n".join(lines) + "\n").encode("utf-8"),
                       capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"mktree 失败: {r.stderr.decode('utf-8', 'ignore')[:200]}")
    got = r.stdout.decode().strip()
    if got != tree_sha:
        raise SystemExit(f"tree 重建后 SHA 不一致：{got[:10]} vs {tree_sha[:10]}")
    print(f"    ↳ 补回 tree {tree_sha[:10]}（{len(lines)} 个文件）")
    return True


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    targets = chain(argv[1]) if argv[0] == "--chain" else argv
    n = 0
    trees = 0
    for sha in targets:
        local = pva.reconstruct_commit(sha)
        ok = "✓" if local == sha else "✗"
        print(f"  {ok} {sha[:10]} → 本地 {local[:10]}")
        meta = pva.api("GET", f"/git/commits/{sha}")
        tree_sha = meta["tree"]["sha"] if isinstance(meta["tree"], dict) else meta["tree"]
        if ensure_tree(tree_sha):
            trees += 1
        n += 1
    print(f"共处理 {n} 个提交，补回 {trees} 个 tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
