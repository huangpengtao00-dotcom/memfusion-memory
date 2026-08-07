# -*- coding: utf-8 -*-
"""诊断 full_input 检索:对每题找出答案相关消息,检查是否被召回及 rank"""
import sys, os, json, re
sys.path.insert(0, "/Users/opall/notes/meshy/3d-ai-learn/aml-challenge/memfusion_v2")
os.environ["MEMFUSION_LLM_BASE_URL"] = "http://127.0.0.1:3001/v1"
os.environ["MEMFUSION_LLM_API_KEY"] = "sk-kZbLUADGmYrspb92Sn5opF2uY7gXyll3CCYWjb2kJywgIXPz"
os.environ["MEMFUSION_FAST_MODEL"] = "batch-cheap"
from wiki_store import WikiStore
from explore_agent import ExploreAgent, LLMDecider
from orchestration import make_orchestrator

KEY = os.environ["MEMFUSION_LLM_API_KEY"]
GW = os.environ["MEMFUSION_LLM_BASE_URL"] + "/chat/completions"

items = json.load(open("/tmp/eval_items_full.json"))
sub = items[:12]

# 每题: 手工标注关键实体(用于在 500 条里定位答案消息)
ENT = {
    "46a3abf7": ["tank", "aquarium", "fish tank", "tanks"],
    "a1cc6108": ["alex", "born"],
    "gpt4_194be4b3": ["instrument", "musical", "piano", "guitar", "ukulele", "saxophone"],
    "28dc39ac": ["game", "gaming", "hours", "played"],
    "gpt4_a56e767c": ["festival", "movie", "film festival"],
    "3fdac837": ["japan", "chicago", "days", "trip"],
    "dfde3500": ["juan", "tutor", "language exchange"],
    "7405e8b1": ["hellofresh", "ubereats", "discount"],
    "55241a1f": ["facebook", "youtube", "comment", "live"],
    "06db6396": ["paint", "project"],
    "7a87bd0c": ["tidying", "tidy", "routine", "daily"],
    "gpt4_4edbafa2": ["bbq", "barbecue"],
}

def parse_roles(messages):
    return messages

for it in sub:
    cid = it["custom_id"]
    store = WikiStore()
    msgs = store.parse_dialog(it["full_input"], max_msgs=10000)
    store.ingest("u", msgs, writer=None, session_id="s0")
    sections = store._collect_sections("u")
    ents = ENT.get(cid, [])
    # 找到含关键实体的消息(答案候选)
    cands = []
    for idx, (d, p, sec) in enumerate(sections):
        c = sec.content.lower()
        hits = [e for e in ents if e in c]
        if hits:
            cands.append((idx, sec.content, hits, sec.role))
    # 运行搜索
    decider = LLMDecider(api_key=KEY, base_url=GW, model="batch-cheap")
    orch = make_orchestrator(max_steps=5, stop_threshold=0.5)
    explorer = ExploreAgent(store, decider=decider, orchestrator=orch)
    res = explorer.explore("u", it["question"], top_k=100)
    top_contents = [r.get("content","") for r in res]
    # 检查答案候选是否在 top100
    print("="*100)
    print(f"[{cid}] {it['dim']} | Q: {it['question'][:70]}")
    print(f"  gold: {str(it['answer'])[:80]}")
    print(f"  total sections: {len(sections)} | search results: {len(res)}")
    for idx, content, hits, role in cands[:8]:
        in_top = content in top_contents
        rank = top_contents.index(content)+1 if in_top else -1
        # 在结果里的排名
        print(f"  {'[HIT]' if in_top else '[miss]'} rank={rank} role={role} hits={hits} :: {content[:100]}")
    if not cands:
        print("  !! 无实体命中消息")
