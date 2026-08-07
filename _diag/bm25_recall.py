# -*- coding: utf-8 -*-
"""BM25-only 检索召回诊断(不跑 embedding,快速)"""
import sys, os, json, re, time
sys.path.insert(0, "/Users/opall/notes/meshy/3d-ai-learn/aml-challenge/memfusion_v2")
os.environ["MEMFUSION_LLM_BASE_URL"] = "http://127.0.0.1:3001/v1"
os.environ["MEMFUSION_LLM_API_KEY"] = "sk-kZbLUADGmYrspb92Sn5opF2uY7gXyll3CCYWjb2kJywgIXPz"
os.environ["MEMFUSION_FAST_MODEL"] = "batch-cheap"
from wiki_store import WikiStore

items = json.load(open("/tmp/eval_items_full.json"))
sub = items[:12]

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

for it in sub:
    cid = it["custom_id"]
    t0=time.time()
    store = WikiStore()
    msgs = store.parse_dialog(it["full_input"], max_msgs=10000)
    store.ingest("u", msgs, writer=None, session_id="s0")
    sections = store._collect_sections("u")
    ents = ENT.get(cid, [])
    cands = []
    for idx, (d, p, sec) in enumerate(sections):
        c = sec.content.lower()
        hits = [e for e in ents if e in c]
        if hits:
            cands.append((idx, sec.content, hits, sec.role))
    # BM25 top_k=100
    res = store.keyword_search("u", it["question"], top_k=100)
    top_contents = [r.get("content","") for r in res]
    print("="*100)
    print(f"[{cid}] {it['dim']} | {it['question'][:55]}")
    print(f"  gold: {str(it['answer'])[:70]} | sections={len(sections)} bm25top={len(res)}")
    for idx, content, hits, role in cands[:10]:
        in_top = content in top_contents
        rank = top_contents.index(content)+1 if in_top else -1
        print(f"  {'[HIT]' if in_top else '[miss]'} bm25rank={rank} role={role} hits={hits} :: {content[:90]}")
    print(f"  bm25 time: {time.time()-t0:.1f}s")
