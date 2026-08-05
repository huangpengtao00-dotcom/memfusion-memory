"""
MemFusion v2 真实数据评测
用 LongMemEval-s 数据集的真实样本，验证 v2 的召回能力。

流程：
1. 拉取真实数据（已下载样例到 /tmp/longmemeval_sample.csv）
2. full_input 作为记忆 → v2 Add
3. question → v2 Search，检查返回证据能否覆盖 answer 关键词
"""
from __future__ import annotations

import csv
import sys
import os
import time
csv.field_size_limit(sys.maxsize)  # LongMemEval 单条很大，加大 csv 限制
sys.path.insert(0, os.path.dirname(__file__))

from wiki_store import WikiStore, Section
from explore_agent import ExploreAgent, LLMDecider
from llm_writer import LLMWriter
from orchestration import make_orchestrator
from llm_config import LLM_API_KEY, LLM_BASE_URL, FAST_MODEL


def load_sample(path="/tmp/longmemeval_sample.csv", n=5):
    """加载前 n 条真实数据。"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            samples.append({
                "question": row["question"],
                "answer": row["answer"],
                "memory": row["full_input"],
            })
    return samples


def parse_messages(memory: str, max_msgs: int = 20) -> list:
    """
    解析 LongMemEval 的 full_input 对话结构。
    full_input 是 "History Chats: Session...: [{'role':..., 'content':...}, ...]" 格式。
    提取每条 user/assistant 消息为独立记忆（比按 chunk 存更能精确召回）。
    """
    import re
    msgs = []
    # 找消息数组：匹配 [{...}, {...}]
    # 简化：按 'role' 切分，提取 content
    # LongMemEval 格式: {'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}
    pattern = r"\{'role':\s*'(\w+)',\s*'content':\s*'((?:[^'\\]|\\.)*)'\}"
    for m in re.finditer(pattern, memory):
        role, content = m.group(1), m.group(2)
        content = content.replace("\\n", "\n").replace("\\'", "'")
        if content.strip():
            msgs.append({"role": role, "content": content})
        if len(msgs) >= max_msgs:
            break
    # 没解析到就用整段（降级）
    if not msgs:
        msgs = [{"role": "user", "content": memory}]
    return msgs


def answer_keywords(answer: str, n=8) -> list:
    """从标准答案提取关键词（用于检查召回覆盖）。"""
    # 取答案里的数字/英文词/中文词，保留短的关键（数字很重要，如 "3"）
    import re
    words = re.findall(r"[a-zA-Z]+", answer)
    nums = re.findall(r"\d+", answer)
    # 英文词取长度>2的，数字全保留
    kws = [w for w in words if len(w) > 2][:n] + nums[:3]
    return kws[:n]


def eval_sample(sample, store, writer, agent, user_id):
    """单条评测：Add 记忆 → Search → 检查覆盖。"""
    # Add: 解析 LongMemEval 的对话结构，按消息建记忆（不是按 chunk）
    # 测试不走 LLM 写入（writer=None），专注测检索召回；真实评测平台会全量 LLM 写
    msgs = parse_messages(sample["memory"], max_msgs=100)
    store.ingest(user_id, msgs, writer=None)  # 无 writer

    # Search
    ev = agent.explore(user_id, sample["question"])
    contents = " ".join(e["content"] for e in ev)

    # 检查 answer 关键词是否在召回里
    kws = answer_keywords(sample["answer"])
    hit = sum(1 for kw in kws if kw.lower() in contents.lower())
    recall = hit / max(len(kws), 1)
    return recall, kws, hit, len(ev)


if __name__ == "__main__":
    samples = load_sample(n=5)
    print(f"加载 {len(samples)} 条真实数据")
    if not samples:
        print("❌ 没有数据，先下载样例")
        sys.exit(1)

    # 构建 v2
    store = WikiStore()
    writer = LLMWriter(api_key=LLM_API_KEY, base_url=LLM_BASE_URL + "/chat/completions", model=FAST_MODEL)
    decider = LLMDecider(api_key=LLM_API_KEY, base_url=LLM_BASE_URL + "/chat/completions", model=FAST_MODEL)
    orch = make_orchestrator(max_steps=5, stop_threshold=0.1)
    agent = ExploreAgent(store, decider=decider, orchestrator=orch)

    recalls = []
    for i, s in enumerate(samples):
        t0 = time.time()
        recall, kws, hit, nev = eval_sample(s, store, writer, agent, f"real:{i}")
        dt = time.time() - t0
        recalls.append(recall)
        print(f"[{i}] recall={recall:.0%} ({hit}/{len(kws)}词) | 召回{nev}条 | {dt:.1f}s")
        print(f"    Q: {s['question'][:50]}")
        print(f"    A: {s['answer'][:40]}")

    print(f"\n=== 平均召回率: {sum(recalls)/len(recalls):.0%} ===")
