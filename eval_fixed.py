"""
MemFusion 固定评测（防过拟合，Fable5 建议）
- 固定 10 条 LongMemEval 数据，统一 parse 100 消息
- 指标：答案关键词在 top-10 证据的覆盖率
- 用于消融：加/去优化对比（去重、窗口、候选池）
"""
from __future__ import annotations

import sys
import csv
import io
import re
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, __file__.rsplit("/", 1)[0])

from wiki_store import WikiStore


def load(n=10):
    with open("/tmp/longmemeval_sample.csv") as f:
        return list(csv.DictReader(io.StringIO(f.read())))[:n]


def recall_one(store, user, q, answer, top_k=10):
    ev = store.hybrid_search(user, q, top_k)
    contents = " ".join(e["content"] for e in ev)
    nums = re.findall(r"\d+", answer)
    words = re.findall(r"[a-zA-Z]{3,}", answer.lower())[:3]
    keys = nums + words
    hit = sum(1 for k in keys if k in contents)
    return hit, len(keys), len(ev)


def run(max_msgs=100, top_k=10):
    rows = load()
    th, tk = 0, 0
    per_q = []
    for i, s in enumerate(rows):
        store = WikiStore()
        msgs = store.parse_dialog(s["full_input"], max_msgs=max_msgs)
        store.ingest(f"u{i}", msgs, writer=None)
        h, k, n = recall_one(store, f"u{i}", s["question"], s["answer"], top_k)
        th += h; tk += k
        per_q.append((h, k))
    return th, tk, per_q


if __name__ == "__main__":
    import time
    t0 = time.time()
    th, tk, per_q = run()
    dt = time.time() - t0
    print(f"固定评测 {len(per_q)} 条, 总召回 {th}/{tk} = {th/max(tk,1):.0%}, {dt:.0f}s")
    for i, (h, k) in enumerate(per_q):
        print(f"  [{i}] {h}/{k}")
