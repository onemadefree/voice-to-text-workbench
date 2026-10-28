#!/usr/bin/env python3
"""等指定的旧进程全部退出后，用新代码续跑剩下的润色。

用途：旧代码的「整批失败就拆半重试」没有上限，服务一挂就把请求量
指数级放大（高太山那篇因此卡了 15 小时零产出）。新代码加了熔断，
但已经跑起来的旧进程改不了，只能等它退出再续跑。

用法：.venv/bin/python scripts/_resume_after_kill.py "pid:文档名" [...]
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PY = str(ROOT / ".venv" / "bin" / "python")
LOGDIR = ROOT / "output"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # 进程还在，只是没权限发信号
        return True
    except OSError:
        return False


def main() -> int:
    targets = []
    for arg in sys.argv[1:]:
        pid_s, _, doc = arg.partition(":")
        if not doc:
            print(f"参数格式应为 pid:文档名，收到 {arg!r}")
            return 1
        targets.append((int(pid_s), doc))

    print(f"等待这些进程退出：{[(p, d) for p, d in targets]}", flush=True)
    while any(alive(pid) for pid, _ in targets):
        time.sleep(10)

    print(f"全部已退出，开始续跑 {time.strftime('%H:%M:%S')}", flush=True)
    for _pid, doc in targets:
        safe = doc.replace("/", "_").replace(":", "：").replace("：", "_")
        log = LOGDIR / f"_resume_{safe}.log"
        # --resume：已润色过的段跳过，只处理剩下的
        with open(log, "ab") as fh:
            subprocess.Popen(
                [PY, str(ROOT / "scripts" / "_local_repolish.py"), "--resume", doc],
                cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"已启动续跑：{doc}（日志 {log.name}）", flush=True)
        time.sleep(3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
