"""
MemFusion v2：AML Add/Search API 服务
用 wiki 记忆 + explore agent 替换骨架的词频检索。
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional

from wiki_store import WikiStore, get_store
from explore_agent import ExploreAgent, LLMDecider
from llm_writer import LLMWriter
from orchestration import make_orchestrator
from llm_config import LLM_BASE_URL, LLM_API_KEY, FAST_MODEL
from entity_extractor import build_count_hint

app = FastAPI(title="MemFusion v2", version="2.6.1")

store: WikiStore = get_store()
decider = LLMDecider(api_key=LLM_API_KEY, base_url=LLM_BASE_URL + "/chat/completions", model=FAST_MODEL)
writer = LLMWriter(api_key=LLM_API_KEY, base_url=LLM_BASE_URL + "/chat/completions", model=FAST_MODEL)
orch = make_orchestrator(max_steps=5, stop_threshold=0.5)
explorer = ExploreAgent(store, decider=decider, orchestrator=orch)


# ---------------- 协议模型（对齐 AML） ----------------

class AddMessage(BaseModel):
    role: str
    timestamp: Optional[int] = None
    content: str


class AddRequest(BaseModel):
    request_id: str
    messages: List[AddMessage]
    user_id: str
    session_id: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    options: Optional[List[str]] = None
    user_id: str
    top_k: int = Field(100, ge=1, le=200)


# ---------------- 接口 ----------------

@app.post("/add")
def add(req: AddRequest):
    """写入记忆到 wiki。LLM 抽事实建页；LLM 挂时降级简单写入。同步完成。
    空 messages（content 为空）→ 过滤掉，仍返回成功（幂等）。
    若 message.content 是"对话历史字符串"（含 History Chats / {'role':...}），
    用 parse_dialog 解析成多条再写入（LongMemEval 等场景）。"""
    msgs = [m.model_dump() for m in req.messages if m.content]
    # 检测"对话历史字符串"：content 含对话消息结构 → 解析成多条
    parsed = []
    for m in msgs:
        c = m["content"]
        if "'role'" in c or "History Chats" in c:
            parsed.extend(store.parse_dialog(c))
        else:
            parsed.append(m)
    # P0(检索侧 agent 端到端验证)：用 writer=None 保留原文，不用 LLM-writer 替换。
    # LLM-writer 把英文原文抽成中文 facts 替换 → answer 级 40→33(-13%)。
    # 评测(USE_LLM_WRITER=0)走 writer=None 得 46，生产必须一致。
    store.ingest(req.user_id, parsed, writer=None, session_id=req.session_id)
    return {
        "success": True,
        "request_id": req.request_id,
        "user_id": req.user_id,
        "session_id": req.session_id,
    }


def format_evidence(r: dict) -> str:
    """
    给证据加元数据头（评分红利：让 answer 模型免猜来源/时间/人物）。
    保留原文，只加简短确定性前缀。元数据缺失则只返回原文。
    """
    content = r.get("content", "")
    parts = []
    src = r.get("source", "")
    ts = r.get("temporal")
    polarity = r.get("polarity", "positive")
    if src:
        parts.append(f"[source: {src}]")
    if ts:
        import datetime
        try:
            dt = datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc).isoformat()
            parts.append(f"[date: {dt}]")
        except Exception:
            pass
    # 极性标记（Fable5：否定是当前有效事实，不是低置信/已失效）
    if polarity == "negative":
        parts.append("[polarity: negative]")
    if parts:
        return " ".join(parts) + "\n" + content
    return content


@app.post("/search")
def search(req: SearchRequest):
    """explore agent 在 wiki 里探索，返回相关证据。只返回证据不生成答案。
    证据 content 带元数据头（source/date），让 answer 模型更容易正确引用。
    用同步 def（FastAPI 自动线程池），避免 async + 阻塞阻塞事件循环。"""
    results = explorer.explore(req.user_id, req.query, top_k=req.top_k)
    # count 类 query：LLM 实体聚簇提示（让 answer 模型能数对），失败降级词法
    try:
        results = build_count_hint(
            req.query, results,
            api_key=LLM_API_KEY, model=FAST_MODEL, base_url=LLM_BASE_URL + "/chat/completions",
        )
    except Exception:
        pass
    # 上送 Current Date（时序题 "X days ago" 参考锚点）已由 explore() 统一注入 as-of 证据，
    # 这里不再重复追加，避免同一条 as-of 出现两次。
    # 限制 top_k
    data = [{
        "id": r.get("id", ""),
        "content": format_evidence(r),
        "score": r.get("score", 0.0),
    } for r in results[:req.top_k]]
    return {"data": data}


@app.get("/health")
async def health():
    return {"status": "ok", "wiki_version": "v2.6.1", "users": len(store.users)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)
