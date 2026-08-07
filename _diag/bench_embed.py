# -*- coding: utf-8 -*-
import sys, time
sys.path.insert(0, "/Users/opall/notes/meshy/3d-ai-learn/aml-challenge/memfusion_v2")
from embedder import get_embedder
em = get_embedder()
docs = ["this is a test message about tanks and aquariums"] * 500
t0 = time.time()
em.embed(docs)
print(f"embed 500 identical msgs: {time.time()-t0:.1f}s")
# realistic: distinct texts
import random
texts = [f"message {i} about topic {random.randint(0,50)} with some words for embedding test" for i in range(500)]
t0 = time.time()
em.embed(texts)
print(f"embed 500 distinct msgs: {time.time()-t0:.1f}s")
