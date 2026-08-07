# -*- coding: utf-8 -*-
"""在 full_input 里定位答案支撑消息(用更强的实体/正则)"""
import sys, os, json, re
sys.path.insert(0, "/Users/opall/notes/meshy/3d-ai-learn/aml-challenge/memfusion_v2")
from wiki_store import WikiStore

items = json.load(open("/tmp/eval_items_full.json"))
sub = {it["custom_id"]: it for it in items[:12]}

# 每题: 答案消息里的关键信号(手工标注)
SIG = {
    "46a3abf7": [r"tank", r"1-gallon", r"friend"],
    "a1cc6108": [r"born", r"birth", r"alex"],
    "gpt4_194be4b3": [r"guitar|piano|instrument|ukulele|saxophone|violin|drum|flute|trumpet"],
    "28dc39ac": [r"\d+\s*hours?\s*(playing|on|of|in)", r"hours playing"],
    "gpt4_a56e767c": [r"festival"],
    "3fdac837": [r"japan|chicago|tokyo|kyoto|osaka"],
}

for cid, pats in SIG.items():
    it = sub[cid]
    store = WikiStore()
    msgs = store.parse_dialog(it["full_input"], max_msgs=10000)
    store.ingest("u", msgs, writer=None, session_id="s0")
    sections = store._collect_sections("u")
    print("="*100)
    print(f"[{cid}] {it['dim']} | Q: {it['question'][:60]} | gold: {str(it['answer'])[:60]}")
    seen=0
    for idx, (d, p, sec) in enumerate(sections):
        c = sec.content
        cl = c.lower()
        if all(any(re.search(pp, cl) for pp in [pats[0], pats[-1]]) for _ in [0]) and any(re.search(pp, cl) for pp in pats):
            # 至少命中一个信号
            print(f"  #{idx} [{sec.role}] {c[:140]}")
            seen += 1
            if seen >= 12: break
    if seen == 0:
        print("  (无信号命中)")
