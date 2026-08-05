"""
MemFusion 评测：v1（词频骨架）vs v2（wiki+explore+语义扩展）

自建模拟评测场景，覆盖 AML 的核心能力维度：
- 事实召回：直接事实
- 多跳组合：需组合两条记忆
- 时序：新旧状态
- 个性化：偏好

评测方式：Add 一批记忆 → 对每个 query 分别用 v1/v2 Search →
检查返回的证据是否包含正确答案（召回率）。
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # 让 v1 memory_server 可 import

from wiki_store import WikiStore, Section
from explore_agent import ExploreAgent, LLMDecider
from llm_writer import LLMWriter
from orchestration import make_orchestrator
from llm_config import LLM_API_KEY, LLM_BASE_URL, FAST_MODEL

# v1 词频（骨架）
from memory_server import MemoryStore


# ---------------- 模拟评测数据 ----------------

MEMORIES = [
    "Alice 喜欢蓝色，讨厌红色",
    "Alice 住在上海浦东新区",
    "Alice 在 Meshy 做软件工程师",
    "Bob 喜欢红色，住在北京",
    "Bob 是一名产品经理",
    "Alice 最近搬到了杭州",
]

QUERIES = [
    {"q": "Alice 喜欢什么颜色", "answer": "蓝色", "dim": "事实召回"},
    {"q": "Alice 住在哪个城市", "answer": "杭州", "dim": "时序（最新状态）"},
    {"q": "Alice 在哪里工作", "answer": "Meshy", "dim": "事实召回"},
    {"q": "Bob 的职业是什么", "answer": "产品经理", "dim": "事实召回"},
    {"q": "Alice 和 Bob 谁喜欢红色", "answer": "Bob", "dim": "多跳组合"},
]


# ---------------- 评测 ----------------

def build_v1():
    ms = MemoryStore()
    for i, m in enumerate(MEMORIES):
        ms.add(f"r{i}", [{"role": "user", "content": m}], "eval:v1")
    return ms


def build_v2():
    store = WikiStore()
    writer = LLMWriter(api_key=LLM_API_KEY, base_url=LLM_BASE_URL + "/chat/completions", model=FAST_MODEL)
    decider = LLMDecider(api_key=LLM_API_KEY, base_url=LLM_BASE_URL + "/chat/completions", model=FAST_MODEL)
    orch = make_orchestrator(max_steps=5, stop_threshold=0.1)
    agent = ExploreAgent(store, decider=decider, orchestrator=orch)
    # 写入（用 writer 抽事实建 wiki）
    store.ingest("eval:v2", [{"role": "user", "content": m} for m in MEMORIES], writer=writer)
    return agent


def hit_rate(system, is_v2=False):
    hits = 0
    for item in QUERIES:
        q, ans, dim = item["q"], item["answer"], item["dim"]
        if is_v2:
            ev = system.explore("eval:v2", q)
        else:
            ev = system.search("eval:v1", q, 10)
        contents = [e["content"] for e in ev]
        hit = any(ans in c for c in contents)
        hits += hit
        print(f"  [{dim}] {q} -> {'✅' if hit else '❌'} 命中'{ans}' | 返回: {contents[:2]}")
    return hits / len(QUERIES)


if __name__ == "__main__":
    print("=" * 50)
    print("v1（词频骨架）")
    print("=" * 50)
    v1 = build_v1()
    r1 = hit_rate(v1, is_v2=False)

    print()
    print("=" * 50)
    print("v2（wiki + explore + 语义扩展）")
    print("=" * 50)
    v2 = build_v2()
    r2 = hit_rate(v2, is_v2=True)

    print()
    print("=" * 50)
    print(f"结果对比:")
    print(f"  v1 召回率: {r1*100:.0f}%")
    print(f"  v2 召回率: {r2*100:.0f}%")
    print(f"  提升: +{(r2-r1)*100:.0f}%")
    print("=" * 50)
