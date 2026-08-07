# -*- coding: utf-8 -*-
"""查看失败题的 focused_input(3条消息)——答案应从这里推出"""
import sys, os, json
sys.path.insert(0, "/Users/opall/notes/meshy/3d-ai-learn/aml-challenge/memfusion_v2")
from wiki_store import WikiStore

items = json.load(open("/tmp/eval_items_full.json"))
sub = {it["custom_id"]: it for it in items[:12]}
fails = ["46a3abf7", "a1cc6108", "gpt4_194be4b3", "28dc39ac", "gpt4_a56e767c"]
for cid in fails:
    it = sub[cid]
    print("="*100)
    print(f"[{cid}] Q: {it['question']}")
    print(f"gold: {it['answer']}")
    print(f"--- focused_input ---")
    print(it["focused_input"][:1800])
