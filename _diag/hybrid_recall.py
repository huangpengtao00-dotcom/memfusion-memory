# -*- coding: utf-8 -*-
"""hybrid(embedding+bm25)召回诊断:答案支撑消息在 top10/top100 的位置"""
import sys, os, json, re, time
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
sub = {it["custom_id"]: it for it in items[:12]}
fails = ["46a3abf7", "a1cc6108", "gpt4_194be4b3", "28dc39ac", "gpt4_a56e767c"]

# 每题答案支撑消息的判别式内容(来自 focused_input,去标点小写后取关键短语)
GOLD_MSGS = {
    "46a3abf7": ["5-gallon tank with a solitary betta fish", "20-gallon freshwater community tank", "anacharis and a java moss", "1-gallon tank"],
    "a1cc6108": ["he's just 21", "turned 32 last month", "just turned 32", "age", "32"],
    "gpt4_194be4b3": ["fender stratocaster", "pearl export", "yamaha fg800", "korg b1", "acoustic guitar"],
    "28dc39ac": ["70 hours", "5 hours", "30 hours", "10 hours", "25 hours"],
    "gpt4_a56e767c": ["austin film festival", "seattle international film festival", "film festival"],
}

for cid in fails:
    it = sub[cid]
    store = WikiStore()
    msgs = store.parse_dialog(it["full_input"], max_msgs=10000)
    store.ingest("u", msgs, writer=None, session_id="s0")
    decider = LLMDecider(api_key=KEY, base_url=GW, model="batch-cheap")
    orch = make_orchestrator(max_steps=5, stop_threshold=0.5)
    explorer = ExploreAgent(store, decider=decider, orchestrator=orch)
    t0 = time.time()
    res = explorer.explore("u", it["question"], top_k=100)
    dt = time.time() - t0
    top10 = [r.get("content","") for r in res[:10]]
    allc = [r.get("content","") for r in res]
    print("="*100)
    print(f"[{cid}] {it['dim']} | {it['question'][:50]} | hybrid time={dt:.0f}s | nres={len(res)}")
    for g in GOLD_MSGS[cid]:
        gl = g.lower()
        # 找含该短语的原始消息在结果里的位置
        hit_in = next((i+1 for i,c in enumerate(allc) if gl in c.lower()), -1)
        hit_top10 = any(gl in c.lower() for c in top10)
        print(f"  goldmsg '{g}': in_top10={hit_top10} overall_rank={hit_in}")
    # 打印 top8
    print("  TOP8:")
    for i, r in enumerate(res[:8]):
        print(f"    #{i} {r.get('content','')[:90]}")
