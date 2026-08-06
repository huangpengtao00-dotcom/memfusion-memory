# MemFusion v2 — Deployment Guide (for platform maintainers)

This document gives everything needed to build, start, and evaluate MemFusion v2
via its Docker entrypoint. Follow it top to bottom — no external services,
private submodules, or undisclosed downloads are required.

## 1. Build

```bash
docker build -t memfusion-v2 .
```

Requires: Docker with a standard Python build environment. No GPU required.
Build time: ~2–4 minutes.

## 2. Run

```bash
docker run -d \
  --name memfusion-v2 \
  -p 8083:8083 \
  -e MEMFUSION_LLM_API_KEY="" \
  memfusion-v2
```

- Service listens on **0.0.0.0:8083** (inside container and host port 8083).
- `MEMFUSION_LLM_API_KEY` is **optional**: if set, LLM-enhanced features
  (semantic expansion, LLM-assisted writing) are enabled; if empty, the system
  degrades gracefully to keyword + vector retrieval (still fully functional
  for Add/Search).

## 3. Health check

```bash
curl http://<host>:8083/health
# → {"status":"ok","wiki_version":"v2","users":0}
```

- Method: `GET /health`
- No authentication required.
- Any 2xx means healthy.

## 4. Add / Search API (AML protocol)

### Add

```
POST /add
```

Request:

```json
{
  "request_id": "eval:run_abc123:locomo_refined:conv-0:chunk-0",
  "messages": [{
    "role": "user",
    "timestamp": 1704067200000,
    "content": "memory text"
  }],
  "user_id": "eval:run_abc123:locomo:conv-0",
  "session_id": "eval:run_abc123:sample:0"
}
```

Response (HTTP 200, synchronous — returns only after memory is stored and searchable):

```json
{
  "success": true,
  "request_id": "eval:run_abc123:locomo_refined:conv-0:chunk-0",
  "user_id": "eval:run_abc123:locomo:conv-0",
  "session_id": "eval:run_abc123:sample:0"
}
```

- `success` is `true`.
- `request_id`, `user_id`, `session_id` are echoed exactly.
- No 202 / task IDs / status polling URLs are returned.

### Search

```
POST /search
```

Request:

```json
{
  "query": "Which answer best matches the memory?",
  "options": ["A. First answer", "B. Second answer"],
  "user_id": "eval:run_abc123:locomo:conv-0",
  "top_k": 100
}
```

Response (HTTP 200):

```json
{
  "data": [{
    "id": "mem_1",
    "content": "remembered fact text",
    "score": 0.87,
    "created_at": "2026-07-01T12:00:00Z"
  }]
}
```

- `data` is an array (empty array if no relevant memory).
- Each item has non-empty `id` and `content`; `score` and `created_at` are optional.
- Results are sorted by relevance (highest first).

## 5. Endpoints summary

| Path | Method | Auth | Purpose |
|---|---|---|---|
| `/health` | GET | None | Health check |
| `/add` | POST | None | Store memory (synchronous) |
| `/search` | POST | None | Retrieve relevant memory |

## 6. Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `MEMFUSION_LLM_API_KEY` | No | empty | Optional LLM key for enhanced features |
| `MEMFUSION_LLM_BASE_URL` | No | https://mx.free.codesonline.dev/v1 | LLM API base URL |
| `MEMFUSION_FAST_MODEL` | No | gpt-4o-mini | LLM model name |

No database, vector store, or external service is required. Everything runs
in-memory within the container.

## 7. Runtime footprint

- CPU: single core is enough
- Memory: ~500MB–1GB
- Disk: ~2GB (image + fastembed model cache)
- Startup: < 30 seconds (first model load may take a bit longer)

## 8. Data handling

Evaluation data is kept in-memory and scoped to `user_id`. No logs of request
bodies are retained. Data is ephemeral (lost on container restart), and no
data leaves the container except to the LLM API (only when
`MEMFUSION_LLM_API_KEY` is set).
