# MemFusion v2

**Wiki-style memory with an explore sub-agent** — a memory system for long-term agents,
with orchestration traces designed for future RL training of the stopping decision.

## What it is

MemFusion v2 is a memory system that:
1. Stores memories as a **wiki** (Dimension → Page → Section + typed links)
2. Retrieves via an **explore sub-agent** (reads the wiki with read-only tools, like code search)
3. Uses **LLM semantic expansion** for lightweight semantic recall (no embedding infra needed)
4. Records **orchestration traces** (spawn/aggregate/stop) with **reward labels** — 
   the exact data shape needed to train the "stopping decision" via RL 
   (a gap named in arXiv:2605.02801).

## Architecture

```
                    AML protocol (Add / Search)
                            │
                            ▼
┌───────────────────────────────────────────────┐
│  api.py           FastAPI: /add /search /health │
├───────────────────────────────────────────────┤
│  explore_agent.py  ExploreAgent + LLMDecider   │
│    - read-only tools: list/browse/read/follow  │
│    - semantic expansion (LLM keywords)         │
│    - decider (LLM or heuristic)                │
├───────────────────────────────────────────────┤
│  orchestration.py  Orchestrator + StopPolicy   │
│    - trace: spawn/aggregate/stop               │
│    - reward label per trace (RL trainable)     │
├───────────────────────────────────────────────┤
│  wiki_store.py    WikiStore                    │
│    - Dimension/Page/Section + typed links      │
│    - keyword_search (zh/en mixed tokenization) │
├───────────────────────────────────────────────┤
│  llm_writer.py    LLMWriter                    │
│    - extract facts → build pages/dimensions    │
└───────────────────────────────────────────────┘
```

## Quickstart

```bash
pip install -r requirements.txt
# configure LLM in llm_config.py (base_url, key, model)
./start.sh            # uvicorn on :8083
# or
python3 -m uvicorn api:app --port 8083 --host 0.0.0.0
```

## API (AML protocol)

```bash
# Add memory
curl -X POST /add -d '{"request_id":"r1","messages":[{"role":"user","content":"用户喜欢蓝色"}],"user_id":"u1","session_id":"s1"}'
# → {"success":true,"request_id":"r1","user_id":"u1","session_id":"s1"}

# Search (explore agent)
curl -X POST /search -d '{"query":"用户喜欢什么颜色","user_id":"u1","top_k":5}'
# → {"data":[{"id":"...","content":"用户喜欢蓝色","score":0.3}]}

# Health
curl /health
# → {"status":"ok","wiki_version":"v2"}
```

## Tests

```bash
python3 test_memfusion.py   # 8 tests: wiki, explore, orchestration, API contract
```

## Design notes

- **Explore sub-agent**: retrieval decoupled into a dedicated agent (like code search),
  not one-shot top-k. Main agent just consumes results.
- **Stopping decision**: the "when to stop" gap named in arXiv:2605.02801 — 
  implemented as a pluggable `StopPolicy` (heuristic now, RL-trainable).
- **Reward-labeled traces**: each explore produces a trace with reward 
  (found evidence=+1, none=-1), exportable as RL training samples.
- **Compliance**: Add/Search use gpt-4o-mini (AML requirement).
- **LLM resilience**: LLM failures degrade to heuristic/keyword search (no empty returns).

## Attribution

- MemCog (arXiv:2605.28046): wiki-style memory structure (Dimension/Page/Section, typed links)
- arXiv:2605.02801: orchestration traces, 5 sub-decisions, "stopping decision" gap
- AML Add/Search protocol: https://agentmemories.ai/api-guide

## Roadmap

- [x] Wiki storage (Dimension/Page/Section + links)
- [x] Explore agent (LLM decider + heuristic fallback)
- [x] Semantic expansion (LLM keywords)
- [x] Orchestration traces + reward labels (RL-trainable)
- [x] AML-compliant (gpt-4o-mini), fast Search (0.3s cached)
- [x] Unit tests, README, requirements, start script
- [ ] Train a stopping policy from collected traces (RL)
- [ ] Multi-step navigation (restore explore browsing, not just keyword recall)

## Docker (AML code submission)

```bash
docker build -t memfusion-v2 .
docker run -p 8083:8083 -e MEMFUSION_LLM_API_KEY=your_key memfusion-v2
```

Endpoints: `POST /add`, `POST /search`, `GET /health`
