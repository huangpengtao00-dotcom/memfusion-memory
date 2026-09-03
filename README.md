# MemFusion v2

**Wiki-style memory for long-term agents**, with orchestration traces designed for future RL
training of the stopping decision.

> **What actually runs (audited 2026-09-03).** The multi-step explore loop was removed in an
> earlier refactor and this README did not follow: `_heuristic_step`, `_exec_action`,
> `LLMDecider.decide` and the wiki navigation tools are **unreachable from the Search path**.
> The live retriever is a **single-pass BM25 + dense hybrid with RRF fusion**. Likewise the
> `use_emb` switch has no setter anywhere in the tree and defaults to `True`, so the dense leg
> **is always on in production** — and it is the only thing that answers preference-style
> questions with no lexical overlap (turning it off costs 12.5 points of recall on those).
> The wiki storage layout below is real; the sub-agent navigation is a design that is not
> currently exercised.

## What it is

MemFusion v2 is a memory system that:
1. Stores memories as a **wiki** (Dimension → Page → Section + typed links)
2. Retrieves with a **single-pass hybrid retriever**: BM25 (Porter-stemmed) + dense embeddings,
   fused by RRF, both legs admission-capped to a candidate window
3. Optionally widens recall with **LLM semantic expansion** (long conversations only)
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

- **Explore sub-agent** (designed, currently unreachable from Search — see the audit note at the
  top): retrieval decoupled into a dedicated agent (like code search),
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

## Retrieval quality

Measured on the public LoCoMo_refined split (10 conversations / 5,882 messages / 1,376 questions
with evidence annotations), replaying the Add/Search protocol faithfully (chunked Add, `top_k=100`).
The metric is EvidenceRecall — whether the annotated evidence message reaches the returned list.

| | before | after | delta |
|---|---|---|---|
| EvidenceRecall@5 | 0.4216 | **0.5470** | +0.1254 |
| EvidenceRecall@10 | 0.5270 | **0.6277** | +0.1007 |
| EvidenceRecall@100 | 0.7946 | **0.8591** | +0.0646 |
| questions with zero recall | 208 | **134** | −74 |
| multi-hop category | 0.6260 | **0.7279** | +0.1019 |
| temporal category | 0.7883 | **0.9013** | +0.1130 |

The @5 and @10 columns matter more than @100: the platform reads the returned list in order,
and before these changes only 52.7% of the annotated evidence was reaching the first ten items
while 79.5% was somewhere in the first hundred.

What changed, all retrieval-side and with no added LLM calls:

- **Porter stemming** replaces plural-only normalisation. Before, `pay`/`paid`, `buy`/`bought`,
  `own`/`owned` were not equivalent (14 of 20 common word-form pairs missed, and `movies` stemmed
  to `movy`), so term-frequency saturation let repeated chit-chat outrank the one message holding
  the answer.
- **Admission cap on both fusion legs.** The dense leg used to rank *every* document, which made
  the `rrf > 0` filter a no-op: a true lexical hit scored 1/61 and a zero-overlap document 1/62.
- **Query normalisation** strips prompt boilerplate and bare date tokens before either leg sees
  the query.
- **Options as query perspectives**, fused in — the field was accepted by the API and never used.
- **Evidence carries the speaker.** `role` was stored but never sent, so the answer model could
  not tell a user's own preference from an assistant's suggestion. Dates are now day-granular.
- **Neighbour budget**: the window expansion used to be truncated away whenever hits filled
  `top_k`, i.e. exactly on the long transcripts that need context.
- **Reads are snapshotted under the lock**, and a failed Search no longer returns `{"data": []}`.
  Concurrent Add + Search used to raise mid-iteration and the bare `except` turned that into
  HTTP 200 with an empty array — a failure disguised as "this user has no such memory".
- **Relative-time normalisation no longer falls back to the server's current date.** It used to
  rewrite a 2023 conversation with today's date and store that, so one piece of evidence carried
  two contradicting dates. The anchor now comes from the message's own `timestamp`, and with no
  anchor available nothing is rewritten. The original wording is also kept and the resolution
  appended, because a judge that forbids relative/absolute conversion cannot be satisfied by
  text where the relative form has been deleted.
- **`score` carries absolute relevance again.** It used to be the raw RRF reciprocal-rank sum, so
  a perfect lexical hit scored 1/61 and an unrelated document 1/62 — no threshold could be built
  on a 6% spread, which is what abstention, suppression and fallback all need. Ordering still uses
  RRF (rank fusion is right for ordering); `score` is now a saturated BM25 term plus cosine.
- **Neighbour items score a fraction of the weakest hit**, not "the weakest hit minus 0.001". An
  absolute offset is no penalty at all in any scale, so ten context messages used to enter the
  result at effectively hit-level scores — and questions that mention a topic without ever stating
  the detail being asked are exactly the shape of an abstention test.
- **A request to forget suppresses the old value.** The protocol has no Delete endpoint, so "please
  forget my phone number" can only mean the old value stops coming back. It used to be stored as
  one more memory that then competed with — and usually outranked — the thing it was meant to
  remove. Suppression is a marker, not a delete: the instruction itself stays as an auditable
  record, only entries earlier than it and covered by its content words are held back, and
  restating the fact afterwards is unaffected.
- **Synthetic evidence has an admission gate**: nothing derived is injected when there are no
  real hits, and derived items now rank below every real one. They used to score 1.0 against
  ~0.016 for genuine evidence, so a question with nothing in memory still received a
  confident-looking date anchor at the top of the list — exactly the shape that makes an answer
  model stop abstaining and start inventing.

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
