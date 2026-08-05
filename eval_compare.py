"""
MemFusion 基准对比：keyword_search（词频） vs hybrid_search（词频+向量 RRF）
用 LongMemEval 真实数据验证哪个检索更好。

指标：含答案消息（含 answer 关键词的消息）在 top-k 里的位置。
位置越靠前越好（评测时答案模型看 top 证据）。
"""
from __future__ import annotations

import sys
import os
import csv
import io
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(__file__))

from wiki_store import WikiStore

SAMPLE = "/tmp/longmemeval_sample.csv"


def load_samples(n=3):
    with open(SAMPLE) as f:
        rows = list(csv.DictReader(io.StringIO(f.read())))
    return rows[:n]


def answer_hit_messages(msgs, answer, question=""):
    """
    找到含答案的消息（ground truth）。
    答案通常是数字（如 "3"），找"数字 + 主题词（clothing/items/pick）同时出现"的消息。
    """
    import re
    nums = re.findall(r"\d+", answer)
    # 从 question 提取主题词
    q_words = re.findall(r"[a-zA-Z]+", question.lower())
    topic = [w for w in q_words if len(w) > 3][:5]  # clothing/items/pick/return
    hit_ids = set()
    for i, m in enumerate(msgs):
        c = m["content"].lower()
        # 答案数字 + 任一主题词同时出现 → 答案消息
        if any(n in c for n in nums) and any(t in c for t in topic):
            hit_ids.add(m.get("id", f"m{i}"))
    return hit_ids


def rank_of_answer(results, hit_ids):
    """答案消息在检索结果里的最前位置（1-indexed，没找到=100）。"""
    for i, r in enumerate(results):
        if r["id"] in hit_ids:
            return i + 1
    return 100


if __name__ == "__main__":
    samples = load_samples()
    print(f"加载 {len(samples)} 条真实数据")

    kw_ranks, hy_ranks = [], []
    for si, s in enumerate(samples):
        store = WikiStore()
        msgs = store.parse_dialog(s["full_input"], max_msgs=100)
        # 记录消息 id（给每条消息一个稳定 id，方便定位答案）
        for i, m in enumerate(msgs):
            m["id"] = f"m{i}"
        store.ingest(f"u{si}", msgs, writer=None)
        hit_ids = answer_hit_messages(msgs, s["answer"])

        q = s["question"]
        kw = store.keyword_search(f"u{si}", q, 10)
        hy = store.hybrid_search(f"u{si}", q, 10)

        kr = rank_of_answer(kw, hit_ids)
        hr = rank_of_answer(hy, hit_ids)
        kw_ranks.append(kr)
        hy_ranks.append(hr)
        print(f"[{si}] Q={q[:40]}... A={s['answer'][:15]}")
        print(f"    词频 rank={kr} | hybrid rank={hr} | {'hybrid更好' if hr < kr else '词频更好' if kr < hr else '打平'}")

    # 汇总
    avg_kw = sum(kw_ranks) / len(kw_ranks)
    avg_hy = sum(hy_ranks) / len(hy_ranks)
    print(f"\n=== 平均答案位置 ===")
    print(f"  词频: {avg_kw:.1f} | hybrid: {avg_hy:.1f}")
    print(f"  结论: {'✅ hybrid 更好' if avg_hy < avg_kw else '❌ 词频更好,需回推'} " if len(samples) > 0 else "")
