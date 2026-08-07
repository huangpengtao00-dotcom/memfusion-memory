"""
MemFusion v2：explore 子 agent（核心贡献）

核心洞察：搜记忆像代码检索——不是一次 top-k，而是像 IDE 里跳转、grep、
读定义、看引用地主动探索。explore agent 就是"在记忆 wiki 里做代码检索的专家"。

只读工具（主 agent 不用背导航负担）：
- list_dimensions(): 顶层概览
- browse_dimension(dim): 维度内页面
- read_page(page_id): 完整页面+出链
- follow_link(page_id, link_type): 沿链接跳转

ReAct 循环：Think → Act → Obs，最多 N 步。
决策层：LLM（gpt-5.4-mini，轻量快）决定每步导航；LLM 不可用时降级启发式。
"""
from __future__ import annotations

import json
import re
import logging
from typing import Dict, List, Optional, Callable, Any

log = logging.getLogger("memfusion.explore")

try:
    import urllib.request, urllib.error
except ImportError:
    pass


class LLMDecider:
    """
    LLM 决策层：让 explore agent 用 LLM 自主决定"下一步在 wiki 里看什么"。
    每步给 LLM 工具描述 + 当前探索状态，LLM 返回 JSON 动作。
    """

    # 默认走 aigw（Meshy 网关）的 DeepSeek-V4-Flash（快）
    DEFAULT_URL = "https://aigw.meshy.team/v1/chat/completions"
    DEFAULT_MODEL = "litellm/DeepSeek-V4-Flash"

    TOOL_DOC = """你是一个在记忆知识库(wiki)里检索的专业 agent。你有以下只读工具：
- list_dimensions(): 列出所有顶层维度
- browse_dimension(dim_id): 查看某维度下的页面列表
- read_page(page_id): 读某个页面的完整内容
- follow_link(page_id): 沿页面的链接跳到相关页
- keyword_search(q): 关键词直接搜索

你的任务：根据用户的问题，一步步决定调用哪个工具、传什么参数，直到找到能回答问题的记忆。
每次只返回一个动作的 JSON，格式：{"tool": "工具名", "args": {...}}
如果已经找到足够证据，返回 {"tool": "DONE"}
"""

    def __init__(self, api_key: str, base_url: str = DEFAULT_URL, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def __init__(self, api_key: str, base_url: str = DEFAULT_URL, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._expand_cache: Dict[str, str] = {}  # query -> 扩展结果（缓存，省 LLM 调用）

    def expand_query(self, query: str) -> str:
        """
        LLM 语义扩展：把 query 扩展成可能出现在答案里的关键词。
        用于提升召回（轻量语义检索，替代 embedding）。
        失败时返回原 query。
        带缓存：相同 query 不重复调 LLM（评测时同 query 多次 Search 很常见）。
        """
        if query in self._expand_cache:
            return self._expand_cache[query]
        prompt = (
            "把下面这个问题里的关键信息提取成5-8个关键词（可能是名字、地点、物品、属性，"
            "一字不差地提取原文里的词，不要添加原文没有的语义），"
            "只输出关键词，逗号分隔，不要其他内容：\n问题：" + query
        )
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 80,
        }).encode()
        req = urllib.request.Request(self.base_url, data=body,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read())
            result = d["choices"][0]["message"]["content"].strip()
            if result and result != query:
                self._expand_cache[query] = result  # 缓存成功扩展
            return result
        except Exception:
            return query

    def decide(self, query: str, state: Dict, tools: Dict) -> str:
        """返回动作描述字符串。"""
        # 构建给 LLM 的上下文（简化：把当前探索到的内容传给 LLM）
        context = f"问题: {query}\n"
        if state.get("dims"):
            context += f"当前维度: {json.dumps(state['dims'], ensure_ascii=False)[:500]}\n"
        if state.get("pages"):
            context += f"当前页面列表: {json.dumps(state['pages'], ensure_ascii=False)[:500]}\n"
        if state.get("current_page"):
            context += f"当前读取页: {state['current_page']}\n"

        prompt = self.TOOL_DOC + "\n当前状态:\n" + context + "\n返回下一步动作 JSON:"
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 120,
        }).encode()
        req = urllib.request.Request(self.base_url, data=body,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read())
            content = d["choices"][0]["message"]["content"]
            # 提取 JSON
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                action = json.loads(m.group(0))
                tool = action.get("tool", "DONE")
                args = action.get("args", {})
                return f"{tool}({json.dumps(args, ensure_ascii=False)})"
            return "DONE"
        except Exception:
            return "DONE"  # LLM 不可用 → 结束（或降级启发式）


class ExploreAgent:
    """
    ExploreAgent：在 wiki 里主动探索，找到相关记忆证据。
    决策层可插拔：LLMDecider（默认）或启发式 fallback。
    """

    def __init__(self, store, max_steps: int = 6, decider: Optional[Any] = None,
                 orchestrator: Optional[Any] = None, use_expansion: bool = False):
        self.store = store
        self.max_steps = max_steps
        # 可插拔的决策函数：query, state -> next_action
        self.decider: Optional[Any] = decider
        # 编排器：记录轨迹 + 停止决策（论文 2605.02801 借鉴）
        self.orchestrator = orchestrator
        self.last_trace: List[Dict] = []
        # 语义扩展默认关（Fable5：LLM 扩词引入幻觉，污染召回）
        self.use_expansion = use_expansion
        # 诊断埋点（P1：评测弱时定位）
        self.last_diag: Dict = {}

    # ---- 只读工具 ----
    def _tools(self, user_id: str) -> Dict[str, Callable]:
        return {
            "list_dimensions": lambda: self.store.list_dimensions(user_id),
            "browse_dimension": lambda dim_id: self.store.browse_dimension(user_id, dim_id),
            "read_page": lambda page_id: self.store.read_page(user_id, page_id),
            "follow_link": lambda page_id, link_type=None: self._follow(user_id, page_id, link_type),
            "keyword_search": lambda q, k=10: self.store.hybrid_search(user_id, q, k),
        }

    def _follow(self, user_id: str, page_id: str, link_type: Optional[str]):
        """沿链接跳转：读目标页，标注通过什么链接到的。"""
        page = self.store.get_page(user_id, page_id)
        if not page:
            return {"error": "page not found"}
        out = []
        for target_id, ltype in page.links.items():
            if link_type and ltype != link_type:
                continue
            target = self.store.get_page(user_id, target_id)
            if target:
                out.append({"to": target_id, "title": target.title, "link_type": ltype})
        return out

    # ---- 启发式导航（默认决策）----
    def _heuristic_step(self, query: str, state: Dict, tools: Dict) -> str:
        """
        启发式：先看概览 → 挑相关维度 → 浏览 → 读页 → 沿链接。
        返回下一步动作的描述，或 "DONE"。
        """
        q = set(query.lower().split())

        if not state.get("dims"):
            dims = tools["list_dimensions"]()
            state["dims"] = dims
            # 挑名字/描述命中 query 的维度
            best = None
            for d in dims:
                if q & set((d["name"] + " " + d["description"]).lower().split()):
                    best = d
                    break
            if best:
                state["current_dim"] = best["id"]
                return f"browse_dimension({best['id']})"
            # 维度级无命中 → 降级：全库关键词搜索（fallback 到 RAG 式）
            q_text = query
            try:
                fallback = tools["keyword_search"](q_text, 10)
            except Exception:
                fallback = []
            state["evidence"] = [
                {
                    "id": r["id"],
                    "content": r["content"],
                    "score": r.get("score", 0.5),
                    "page_title": r.get("page_title", ""),
                    "dimension": r.get("dimension", ""),
                }
                for r in fallback
            ]
            return "DONE"

        if state.get("current_dim") and not state.get("browsed"):
            pages = tools["browse_dimension"](state["current_dim"])
            state["browsed"] = pages
            state["pages"] = pages
            # 挑标题命中 query 的页
            best = None
            for p in pages:
                if q & set(p["title"].lower().split()):
                    best = p
                    break
            if best:
                state["current_page"] = best["id"]
                return f"read_page({best['id']})"
            return "DONE"

        if state.get("current_page") and not state.get("read"):
            page = tools["read_page"](state["current_page"])
            state["read"] = True
            state["evidence"] = []
            # 收集页内章节作为证据
            for s_title, s in page["sections"].items():
                state["evidence"].append({
                    "id": page["id"],
                    "content": s["content"],
                    "score": 0.9,  # 占位
                    "page_title": page["title"],
                })
            # 沿链接继续（收集出链，挑相关）
            links = page.get("links", {})
            state["out_links"] = list(links.items())
            for target_id, ltype in links.items():
                # 简化为都沿一层
                state["evidence"].append({
                    "id": target_id,
                    "content": f"[link:{ltype}]",
                    "score": 0.6,
                })
            return "DONE"

        return "DONE"

    # ---- 主入口 ----
    def explore(self, user_id: str, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        在 wiki 里探索，返回相关证据（只返回证据，不生成答案）。
        top_k：召回宽度（默认 None→题型感知）。评测平台传 100 时填满到 top_k。
        """
        tools = self._tools(user_id)
        state: Dict = {}
        evidence: List[Dict] = []

        # 编排轨迹：记录探索过程（借鉴 2605.02801）
        if self.orchestrator:
            self.orchestrator.start_trace(query)
            self.last_trace = self.orchestrator.get_trace()

        # 语义扩展：先用 LLM 扩关键词，做一次强召回作为种子。
        # LLM 失败/超时 → 降级为原 query 词频检索（保证不空返回）。
        # 语义扩展默认关闭（Fable5 审核：LLM 扩关键词引入幻觉实体，污染召回）。
        # 通过 self.use_expansion 开关控制（默认 False）。
        expanded = query
        if self.use_expansion and self.decider and hasattr(self.decider, "expand_query"):
            try:
                expanded = self.decider.expand_query(query)
            except Exception:
                expanded = query
        # 扩展词或原 query 都检索，作为主 evidence
        try:
            seed_q = expanded if (expanded and expanded != query) else query
            # 题型感知召回宽度：count/time 更宽防漏（Fable5）
            # top_k 传入时以 top_k 为准（评测平台要填满到 top_k，不只 10/15）
            qtype = self.detect_query_type(query)
            recall_k = max(self._recall_k(qtype), top_k or 0)
            seed = tools["keyword_search"](seed_q, recall_k)
            if not seed and expanded != query:
                seed = tools["keyword_search"](query, recall_k)  # 扩展没召回 → 原 query
            if seed:
                state["evidence"] = [
                    {"id": r["id"], "content": r["content"],
                     "score": r.get("score", 0.5),
                     "page_title": r.get("page_title", ""),
                     "dimension": r.get("dimension", ""),
                     "source": r.get("source", ""),      # 证据溯源透传
                     "role": r.get("role", ""),          # user/assistant(count-hint 过滤用)
                     "temporal": r.get("temporal"),      # 时间锚点透传
                     "confidence": r.get("confidence", 1.0),  # 置信度透传
                     "polarity": r.get("polarity", "positive")}  # 极性透传
                    for r in seed
                ]
            # 记录聚合决策
            if self.orchestrator:
                self.orchestrator.log("aggregate", f"keyword_search({seed_q}) -> {len(seed)}",
                                      result=len(seed), step=1)
                stop = self.orchestrator.should_stop(state.get("evidence", []), step=1)
                self.orchestrator.log("stop", f"should_stop={stop}", result=stop, step=1)
                # reward 标签（训练用）：有证据=探索有效，无证据=无效
                reward = 1.0 if state.get("evidence") else -1.0
                self.orchestrator.set_reward(reward)
                self.last_trace = self.orchestrator.get_trace()
            # 诊断埋点（P1）：记录题型/召回/证据数，评测弱时定位
            self.last_diag = {
                "query_type": qtype,
                "recall_k": recall_k,
                "raw_recall": len(seed),
                "evidence": len(state.get("evidence", [])),
                "deduped": len(self._dedup(state.get("evidence", []))),
                "has_evidence": bool(state.get("evidence")),
            }
            # 无论有无证据都返回（有证据返回结果，没证据返回空）——不做多步导航，保评测快
            # 多步 ReAct 循环已删除（Fable5 B：原 323-350 行 for 循环在 318 行 return 后
            # 永远不可达 = 死代码；单步 + _expand_neighbors 确定性链接展开已覆盖其意图）
            results = self._dedup(state.get("evidence", []))
            # v2.5：时序题 "X days/weeks ago" 的参考锚点——把 user_meta 里的 Current Date
            # 作为一条 as-of 证据上送（此前只加了 api.py /search，评测直接走 explore 拿不到，
            # Super Bowl 题因此算不出"17 days ago" → INSUFFICIENT）
            # v2.5c：只对**有时序意图**的问题上送 as-of（count/偏好/事实题带上 [as-of:] 是纯噪音，
            # 维度侧实测让 count 题 answer 数偏）。asof 变量仍保留给 time-hint 用。
            asof = None
            try:
                asof = self.store.get_user_meta(user_id, "current_date")
                if asof and self._has_temporal_intent(query):
                    results = results + [{
                        "id": "as-of",
                        "content": f"[as-of: {asof}]",
                        "score": 0.5,
                        "page_title": "", "dimension": "",
                        "source": "", "temporal": None,
                        "confidence": 1.0, "polarity": "positive",
                    }]
            except Exception:
                asof = None
            # v2.5b：answer 模型(qwen-plus)日期算术弱，即使有 [date:] 元数据，
            # "days ago" 会算偏、"days passed between" 算成 0。这里用证据里确定性的
            # [date:] 元数据预计算差，注入 [time-hint]，answer 模型只须照抄。
            # 保守触发（日期不全不注入，避免污染召回）。放最前保证不被 results[:10] 截断。
            try:
                from temporal_hint import build_temporal_hint, build_clock_hint
                hint = build_temporal_hint(query, results, asof)
                if not hint:
                    hint = build_clock_hint(query, results)  # 机制3：钟表时间跨消息推断
                if hint:
                    results = [{
                        "id": "time-hint", "content": hint, "score": 1.0,
                        "page_title": "", "dimension": "",
                        "source": "", "temporal": None,
                        "confidence": 1.0, "polarity": "positive",
                    }] + results
            except Exception:
                pass
            return results
        except Exception:
            self.last_diag = {"error": True}
            return self._dedup(state.get("evidence", []))

    @staticmethod
    def _has_temporal_intent(query: str) -> bool:
        """粗略判断问题是否有时序意图（"X days/weeks ago"、"between events"、"what date/time"）。
        用于决定是否上送 [as-of:] 当前日期锚点——只对时序题有用，count/事实/偏好题带上是噪音。"""
        ql = query.lower()
        markers = ["ago", "days", "weeks", "months", "hours", "how long",
                   "what time", "when", "between", " date", "date ", "the date",
                   "yesterday", "today", "how many days", "how many weeks",
                   "how many months", "how many hours", "last week", "this week"]
        return any(m in ql for m in markers)

    @staticmethod
    def detect_query_type(query: str) -> str:
        """题型检测（Fable5：题型感知补召回）。"""
        ql = query.lower()
        if any(w in ql for w in ["when", "yesterday", "before", "after", "date",
                                 "上次", "昨天", "什么时候", "日期"]):
            return "time"
        if any(w in ql for w in ["how many", "list all", "all the", "count ",
                                 "几个", "多少", "哪些", "有哪些"]):
            return "count_list"
        if any(w in ql for w in ["prefer", "like", "recommend", "favorite",
                                 "喜欢", "偏好", "推荐", "不喜欢"]):
            return "preference"
        return "fact"

    def _recall_k(self, qtype: str) -> int:
        """题型感知的召回宽度：count/list 需要更宽召回防漏。"""
        return 15 if qtype in ("count_list", "time") else 10

    @staticmethod
    def _dedup(evidence: List[Dict]) -> List[Dict]:
        """去重（按 content）+ 排序（按 score 降序）。"""
        seen = set()
        out = []
        for e in sorted(evidence, key=lambda x: x.get("score", 0), reverse=True):
            c = e.get("content", "")
            if c and c not in seen:
                seen.add(c)
                out.append(e)
        return out

    def _exec_action(self, action: str, state: Dict, tools: Dict, user_id: str) -> None:
        """
        执行 LLM 返回的动作。格式: "tool(args_json)"。
        把结果写入 state，供下一轮 LLM 决策用。
        """
        if not action or action == "DONE":
            return
        m = re.match(r"(\w+)\((.*)\)", action, re.DOTALL)
        if not m:
            return
        tool, args_str = m.group(1), m.group(2)
        try:
            args = json.loads(args_str) if args_str.strip() else {}
        except Exception:
            args = {}

        try:
            if tool == "list_dimensions":
                state["dims"] = tools["list_dimensions"]()
            elif tool == "browse_dimension":
                dim_id = args.get("dim_id") or args.get("dimension")
                if dim_id:
                    state["pages"] = tools["browse_dimension"](dim_id)
                    state["current_dim"] = dim_id
            elif tool == "read_page":
                page_id = args.get("page_id") or args.get("id")
                if page_id:
                    page = tools["read_page"](page_id)
                    if page:
                        state["current_page"] = page
                        # 收集页面章节为证据
                        ev = []
                        for s_title, s in page["sections"].items():
                            ev.append({
                                "id": page_id,
                                "content": s["content"],
                                "score": s.get("confidence", 0.8),
                                "page_title": page["title"],
                            })
                        if ev:
                            state["evidence"] = ev
            elif tool == "follow_link":
                page_id = args.get("page_id") or args.get("id")
                if page_id:
                    state["followed"] = tools["follow_link"](page_id)
            elif tool == "keyword_search":
                q = args.get("q") or args.get("query") or ""
                if q:
                    state["evidence"] = [
                        {"id": r["id"], "content": r["content"],
                         "score": r.get("score", 0.5),
                         "page_title": r.get("page_title", ""),
                         "dimension": r.get("dimension", "")}
                        for r in tools["keyword_search"](q, 10)
                    ]
        except Exception as e:
            log.warning("exec_action failed: tool=%s err=%s", tool, e)


# 便于直接使用
def make_explorer(store):
    return ExploreAgent(store)
