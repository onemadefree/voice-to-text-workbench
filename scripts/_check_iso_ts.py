"""校验 push_via_api.ident_to_iso 的时间戳换算是否自洽。

要求：ISO 串（墙钟+偏移）还原出的 epoch 必须等于原始 ts。
"""
import importlib.util
import datetime
import subprocess
import sys

spec = importlib.util.spec_from_file_location("pva", "scripts/push_via_api.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

ok = True
for ref in ["HEAD", "refs/heads/research/asr-cross-platform", "v1.0.0^{commit}"]:
    raw = subprocess.run(["git", "cat-file", "commit", ref],
                         capture_output=True, text=True).stdout
    for line in raw.splitlines():
        if line.startswith(("author ", "committer ")):
            key, _, val = line.partition(" ")
            d = m.ident_to_iso(val)
            back = int(datetime.datetime.strptime(
                d["date"], "%Y-%m-%dT%H:%M:%S%z").timestamp())
            good = back == d["ts"]
            ok &= good
            print(f"{'OK ' if good else 'BAD'} {ref:38s} {key:9s} "
                  f"ts={d['ts']} iso={d['date']} 还原={back}")

print("\n结论：", "全部自洽" if ok else "仍有偏移")
sys.exit(0 if ok else 1)
