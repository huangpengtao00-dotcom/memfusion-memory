"""
MemFusion v2：wiki 记忆存储层（数据模型 + 增量写入）

结构：Dimension → Page → Section，四类类型化链接。
借鉴 MemCog，但作为独立可测试的存储模块实现。

链接类型：
- related_to   语义关联
- temporal_next  时间承接
- caused_by    因果
- contrasts_with 对比
"""
from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set


# ---------------- 数据模型 ----------------

@dataclass
class Section:
    """页面内的章节，细粒度信息块。
    带证据溯源（可审计记忆）：source 记录记忆来自哪条对话/消息。"""
    title: str
    content: str                       # 记忆文本
    facts: List[str] = field(default_factory=list)   # 结构化事实
    related: List[str] = field(default_factory=list)  # 关联页面 id
    temporal: Optional[float] = None   # 时间上下文
    confidence: float = 1.0            # 抽取置信度（≠极性，≠状态）
    polarity: str = "positive"         # positive / negative（Fable5：否定≠低置信）
    source: str = ""                   # 来源（对话/消息 id，可审计）
    order: int = 0                     # 对话内消息序号（滑动窗口用）
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Page:
    """覆盖一个连贯主题的知识单元。"""
    id: str
    title: str
    summary: str = ""
    sections: Dict[str, Section] = field(default_factory=dict)  # title -> Section
    links: Dict[str, str] = field(default_factory=dict)  # target_page_id -> link_type
    created_at: float = field(default_factory=time.time)

    def add_section(self, section: Section) -> None:
        self.sections[section.title] = section

    def add_link(self, target_id: str, link_type: str) -> None:
        self.links[target_id] = link_type

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "sections": {k: v.to_dict() for k, v in self.sections.items()},
            "links": self.links,
            "created_at": self.created_at,
        }


@dataclass
class Dimension:
    """顶层分类。"""
    id: str
    name: str
    description: str = ""
    pages: Dict[str, Page] = field(default_factory=dict)  # page_id -> Page

    def add_page(self, page: Page) -> None:
        self.pages[page.id] = page


# ---------------- 存储层 ----------------

class WikiStore:
    """按 user_id 隔离的 wiki 记忆库。"""

    LINK_TYPES = ("related_to", "temporal_next", "caused_by", "contrasts_with")

    def __init__(self):
        # user_id -> {dim_id -> Dimension}
        self.users: Dict[str, Dict[str, Dimension]] = {}
        # 线程锁：FastAPI 多线程并发 Add/Search，写操作需要保护
        self._lock = __import__("threading").RLock()

    # ---- user 管理 ----
    def _ensure_user(self, user_id: str) -> Dict[str, Dimension]:
        with self._lock:
            if user_id not in self.users:
                self.users[user_id] = {}
            return self.users[user_id]

    # ---- 维度操作 ----
    def add_dimension(self, user_id: str, name: str, description: str = "") -> Dimension:
        dims = self._ensure_user(user_id)
        dim = Dimension(id=str(uuid.uuid4())[:8], name=name, description=description)
        dims[dim.id] = dim
        return dim

    def list_dimensions(self, user_id: str) -> List[Dict]:
        dims = self._ensure_user(user_id)
        return [
            {"id": d.id, "name": d.name, "description": d.description, "n_pages": len(d.pages)}
            for d in dims.values()
        ]

    # ---- 页面操作 ----
    def add_page(self, user_id: str, dim_id: str, title: str, summary: str = "") -> Optional[Page]:
        dims = self._ensure_user(user_id)
        dim = dims.get(dim_id)
        if not dim:
            return None
        page = Page(id=str(uuid.uuid4())[:8], title=title, summary=summary)
        dim.add_page(page)
        return page

    def get_page(self, user_id: str, page_id: str) -> Optional[Page]:
        dims = self._ensure_user(user_id)
        for dim in dims.values():
            if page_id in dim.pages:
                return dim.pages[page_id]
        return None

    def browse_dimension(self, user_id: str, dim_id: str) -> List[Dict]:
        dims = self._ensure_user(user_id)
        dim = dims.get(dim_id)
        if not dim:
            return []
        return [
            {"id": p.id, "title": p.title, "summary": p.summary,
             "n_sections": len(p.sections), "links": list(p.links.keys())}
            for p in dim.pages.values()
        ]

    def read_page(self, user_id: str, page_id: str) -> Optional[Dict]:
        page = self.get_page(user_id, page_id)
        return page.to_dict() if page else None

    # ---- 写入：消息 → wiki ----
    def ingest(self, user_id: str, messages: List[Dict], writer=None,
               session_id: Optional[str] = None) -> int:
        """
        把一批消息写入 wiki。
        writer 提供时：LLM 抽事实 → 按 topic 建页/更新 → 按 dimension 归维度。
        writer 为 None 时：降级为简单匹配（每条消息建/更新一页）。
        session_id：无 source 的消息用它兜底分组（_expand_neighbors 依赖）。

        建链接逻辑：
        - 同一批消息抽出的 facts，如果 topic 不同但有重叠实体 → related_to
        - temporal 相邻的页面 → temporal_next（简化：同批页都算 temporal_next）
        """
        # 整批写入加锁：并发 Add 时保证原子性
        with self._lock:
            return self._ingest_unlocked(user_id, messages, writer, session_id)

    def _ingest_unlocked(self, user_id: str, messages: List[Dict], writer=None,
                         session_id: Optional[str] = None) -> int:
        """加锁内的实际写入逻辑。"""
        dims = self._ensure_user(user_id)
        written = 0
        pages_this_batch: List[Page] = []

        # 提取会话日期作为相对时间归一化锚点（bug 修复：不用 today()）
        # 优先取 parse_dialog 附到消息上的 session_date（真实 LongMemEval 日期在
        # full_input 头部，不在消息 content 里）；构造消息没有则回落到内容扫描。
        session_date = None
        try:
            for msg in messages:
                sd = msg.get("session_date")
                if sd:
                    session_date = sd
                    break
            if session_date is None:
                from time_utils import extract_session_date
                for msg in messages:
                    sd = extract_session_date(msg.get("content", ""))
                    if sd:
                        session_date = sd
                        break
        except Exception:
            pass

        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", "")
            if not content:
                continue

            # source 兜底：_expand_neighbors 按 source 分组判相邻，空 source 会全失效
            if not msg.get("source"):
                msg["source"] = session_id or f"{user_id}:batch"

            # 相对时间归一化（用会话日期当锚点，不用 today()）
            try:
                from time_utils import normalize_relative_times
                content = normalize_relative_times(content, ref_date=session_date)
            except Exception:
                pass

            # 否定/修正检测（Fable5：personamem 偏好更新关键）
            is_negation = self._is_negation(content)

            # 用 LLM 抽取（writer 提供时）
            info = None
            if writer:
                try:
                    info = writer.extract(content)
                except Exception:
                    info = None

            if info:
                topic = info.get("topic") or content[:30]
                facts = info.get("facts") or [content]
                dim_name = info.get("dimension") or "General"

                # 找/建维度
                target_dim = None
                for d in dims.values():
                    if d.name == dim_name:
                        target_dim = d
                        break
                if not target_dim:
                    target_dim = self.add_dimension(user_id, dim_name, f"{dim_name} 相关记忆")
                    dims = self._ensure_user(user_id)

                # 找/建页面（按 topic 匹配）
                page = None
                for p in target_dim.pages.values():
                    if p.title == topic:
                        page = p
                        break
                if not page:
                    page = self.add_page(user_id, target_dim.id, topic, topic)
                    if page:
                        pages_this_batch.append(page)
                if page:
                    # 添加结构化 facts 作为 sections
                    for fact in facts:
                        page.add_section(Section(
                            title=fact[:30],
                            content=fact,
                            temporal=msg.get("timestamp"),
                            # 否定是极性标记，不影响置信度（Fable5 修正）
                            polarity="negative" if is_negation else "positive",
                            confidence=0.85,
                            source=msg.get("source", ""),  # 证据溯源
                            order=msg_idx,                  # 消息序号
                        ))
            else:
                # 降级：简单建页
                if not dims:
                    dim = self.add_dimension(user_id, "General", "General knowledge")
                else:
                    dim = next(iter(dims.values()))
                page = self.add_page(user_id, dim.id, content[:30], content[:60])
                if page:
                    page.add_section(Section(
                        title=content[:30], content=content,
                        temporal=msg.get("timestamp"),
                        # 否定是极性标记，不影响置信度
                        polarity="negative" if is_negation else "positive",
                        confidence=0.8,
                        source=msg.get("source", ""),  # 证据溯源（窗口扩展分组键）
                        order=msg_idx,  # 消息序号（滑动窗口用）
                    ))
                    pages_this_batch.append(page)
            written += 1

        # 建链接：同批页面互相 related_to
        for i in range(len(pages_this_batch)):
            for j in range(i + 1, len(pages_this_batch)):
                a, b = pages_this_batch[i], pages_this_batch[j]
                if a.id not in b.links:
                    b.add_link(a.id, "related_to")
                if b.id not in a.links:
                    a.add_link(b.id, "related_to")
        return written

    # ---- 简单检索（后续 explore agent 用）----
    def keyword_search(self, user_id: str, query: str, top_k: int = 10) -> List[Dict]:
        """关键词匹配页面/章节（占位，explore agent 阶段换 embedding）。"""
        q = self._tokenize(query)
        results = []
        dims = self._ensure_user(user_id)
        now = time.time()
        for dim in dims.values():
            for page in dim.pages.values():
                for section in page.sections.values():
                    score = self._overlap(q, section.content)
                    if score > 0:
                        results.append({
                            "id": page.id,
                            "content": section.content,
                            "score": round(score, 4),
                            "page_title": page.title,
                            "dimension": dim.name,
                            "created_at": section.temporal,
                        })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def hybrid_search(self, user_id: str, query: str, top_k: int = 10) -> List[Dict]:
        """
        混合检索：词频 + 语义向量，RRF 融合排序（借鉴 Mem0 / Engram）。
        补词频抓不住的语义关联（如 query "items of clothing" ↔ "pick up dry cleaning"）。
        embedding 不可用时降级纯词频。
        """
        # 1. 词频检索（keyword_search 已有）
        kw = self.keyword_search(user_id, query, top_k=top_k)
        kw_ids = [r["id"] for r in kw]
        kw_rank = {rid: i + 1 for i, rid in enumerate(kw_ids)}

        # 2. 语义向量检索
        dims = self._ensure_user(user_id)
        memories = []
        mem_map = {}  # id -> section
        for dim in dims.values():
            for page in dim.pages.values():
                for section in page.sections.values():
                    memories.append(section.content)
                    mem_map[section.content] = {
                        "id": page.id,
                        "content": section.content,
                        "score": 0.0,
                        "page_title": page.title,
                        "dimension": dim.name,
                        "source": section.source,       # 证据溯源
                        "temporal": section.temporal,   # 时间锚点
                        "confidence": section.confidence,  # 抽取置信度
                        "polarity": section.polarity,   # 极性（Fable5：否定≠低置信）
                        "order": section.order,         # 消息序号（滑动窗口用）
                        "speaker": section.facts[0] if section.facts else "",  # speaker 占位
                    }

        emb_scores = {}
        if memories:
            from embedder import get_embedder
            scores = get_embedder().search(query, memories, top_k=top_k)
            # 按相似度排序取 rank（content -> rank）
            order = sorted(range(len(memories)), key=lambda i: -scores[i])
            for rank, idx in enumerate(order):
                c = memories[idx]
                emb_scores[c] = rank + 1
                mem_map[c]["score"] = round(float(scores[idx]), 4)

        # 3. RRF 融合：score = sum(1/(k + rank))，k=60
        K = 60
        rrf = {}
        for c in mem_map:
            s = 0.0
            if c in kw_rank:
                s += 1.0 / (K + kw_rank[c])
            if c in emb_scores:  # 语义 rank（content -> rank）
                s += 1.0 / (K + emb_scores[c])
            rrf[c] = s

        # 按 RRF 排序
        sorted_content = sorted(rrf, key=rrf.get, reverse=True)
        results = [mem_map[c] for c in sorted_content[:top_k]]
        # 只保留"至少被词频或语义命中"的（RRF>0）
        results = [r for r in results if rrf.get(r["content"], 0) > 0]
        # 滑动窗口扩展：命中消息的相邻消息也补召回（对话连续性，locomo/scriptmem 事件簇）
        results = self._expand_neighbors(user_id, results, window=2, max_extra=10)
        # count 聚簇提示由 api.py 的 LLM 实体提取(build_count_hint)负责，这里不重复
        # 注意：近重复去重(_dedup_similar) 未验证出提升，暂不启用（避免未验证改动）
        return results[:top_k]

    @staticmethod
    def _cluster_count_hint(query: str, results: List[Dict],
                            top_k: int = 100) -> List[Dict]:
        """
        count 类题目聚簇提示：统计召回消息里高频出现的"主题实体"（词或短语），
        对每个高频主题标注"提到 N 次"（N=包含该实体的不同消息数）。
        追加为一条结构化证据，让 answer 模型能数出 count 题答案。
        仅当 query 是 count 类（how many / how much / count）才触发。
        词法近似：不保证语义完全正确，但给 answer 一个可数的线索。
        """
        # 只在 count 类 query 触发（避免普通题被聚簇噪音干扰）
        if not any(w in query.lower() for w in ["how many", "how much", "count", "number of", "几", "多少", "几个"]):
            return results

        # 统计召回内容里出现的高频词（长度>3 的英文词），排除 stopword
        from collections import Counter
        import re
        stop = {"that", "this", "with", "have", "from", "they", "there", "what",
                "when", "your", "about", "some", "these", "those", "them", "been",
                "were", "will", "would", "could", "should", "because", "then", "than",
                "really", "very", "just", "make", "made", "think", "need", "going",
                "want", "like", "know", "one", "well", "even", "still", "also", "back"}
        counter = Counter()
        for r in results:
            content = r.get("content", "")
            words = [w for w in re.findall(r"[a-z]{4,}", content.lower()) if w not in stop]
            counter.update(set(words))  # 每条消息内去重，跨消息计数

        # 高频主题实体（出现在 >=2 条不同消息）
        themes = [w for w, c in counter.items() if c >= 2]
        themes = themes[:5]  # 最多 5 个主题

        if not themes:
            return results

        # 每个主题：统计包含它的不同消息数
        hint_parts = []
        for theme in themes:
            msg_count = sum(1 for r in results if re.search(rf"\b{re.escape(theme)}\b",
                                                            r.get("content", "").lower()))
            hint_parts.append(f"{theme}:{msg_count}")
        hint = "[count-hint] " + ", ".join(hint_parts)
        results = results + [{
            "id": "count-hint", "content": hint, "score": 0.5,
            "page_title": "", "dimension": "", "source": "", "order": -1,
        }]
        return results

    def _expand_neighbors(self, user_id: str, results: List[Dict],
                          window: int = 2, max_extra: int = 10) -> List[Dict]:
        """
        滑动窗口扩展：对已命中消息，补召回同 source 对话里相邻 window 条消息。
        解决"答案依赖上下文连续消息"（locomo 事件簇 / scriptmem 场景）。
        """
        if not results:
            return results
        # 构建 (source, order) -> section 索引（按对话分组）
        from collections import defaultdict
        by_source = defaultdict(list)  # source -> [(order, content, page_id)]
        dims = self._ensure_user(user_id)
        for dim in dims.values():
            for page in dim.pages.values():
                for section in page.sections.values():
                    if section.source:
                        by_source[section.source].append(
                            (section.order, section.content, page.id, section.source))

        # 对每个 source 按 order 排序
        for src in by_source:
            by_source[src].sort()

        # 收集命中消息的 source+order，补相邻
        hit_keys = set()
        for r in results:
            if r.get("source"):
                hit_keys.add((r["source"], r.get("order", 0)))

        extra = []
        seen = {r["content"] for r in results}
        for src, orders in by_source.items():
            orders_list = sorted(orders)
            for (order, content, page_id, source) in orders_list:
                if (src, order) in hit_keys:
                    # 命中 → 补相邻
                    continue
                # 检查这条是否在某个命中消息的 window 内
                for (h_src, h_order) in hit_keys:
                    if h_src == src and abs(order - h_order) <= window:
                        if content not in seen and len(extra) < max_extra:
                            extra.append({
                                "id": page_id, "content": content, "score": 0.3,
                                "source": source, "order": order,
                                "page_title": "", "dimension": "",
                            })
                            seen.add(content)
                        break
        return results + extra

    @staticmethod
    def _dedup_similar(results: List[Dict], sim_thresh: float = 0.7) -> List[Dict]:
        """
        近重复证据去重（Fable5：count 题防同一事件重复占位）。
        按 token 重叠度合并高度相似的证据，保留分数最高的一条。
        这样 top-k 能容纳更多独立事件（doctors/festivals 计数）。
        """
        if not results:
            return results
        deduped = []
        for r in sorted(results, key=lambda x: -x.get("score", 0)):
            r_tokens = set(WikiStore._tokenize(r["content"]))
            if not r_tokens:
                deduped.append(r)
                continue
            is_dup = False
            for d in deduped:
                d_tokens = set(WikiStore._tokenize(d["content"]))
                if not d_tokens:
                    continue
                inter = len(r_tokens & d_tokens)
                sim = inter / min(len(r_tokens), len(d_tokens))  # 最小覆盖率
                if sim >= sim_thresh:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(r)
        return deduped

    @staticmethod
    def _is_negation(text: str) -> bool:
        """否定/修正表达检测（personamem 偏好更新关键）。"""
        t = text.lower()
        neg = ["不再喜欢", "不喜欢", "no longer", "actually", "instead",
               "changed my mind", "改为", "改成", "其实", "not "]
        return any(n in t for n in neg)

    @staticmethod
    def parse_dialog(memory: str, max_msgs: int = 100) -> List[Dict]:
        """
        解析对话历史（LongMemEval 等格式）为消息列表。
        full_input 是 "History Chats: ... [{'role':.., 'content':..}, ...]" 字符串。
        按消息建记忆（比按 chunk 更能精确召回答案所在的消息）。
        """
        import re
        # 会话日期在 full_input 头部（"History Chats:Session 2023/02/15 ..."），
        # 不在消息 content 里——先提取，后面附到每条消息上，ingest 才能用对锚点
        session_date = None
        try:
            from time_utils import extract_session_date
            session_date = extract_session_date(memory)
        except Exception:
            pass
        # 会话标识：从头部提取 Session YYYY/MM/DD 片段作为同源分组键
        # （_expand_neighbors 按 source 分组判相邻，无 source 时窗口扩展完全失效）
        source = "dialog"
        try:
            m = re.search(r"History Chats:\s*(Session[^:]*:?)", memory)
            if m:
                source = m.group(1).strip(" :")
        except Exception:
            pass

        msgs = []
        # content 实际用双引号包裹（如 "I'm making progress..."），单引号正则会漏解析。
        # 兼容双引号 content；role 用单引号。
        pattern = r"\{'role':\s*'(\w+)',\s*'content':\s*\"((?:[^\"\\]|\\.)*)\"\}"
        for m in re.finditer(pattern, memory):
            role, content = m.group(1), m.group(2)
            content = content.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
            if content.strip():
                msgs.append({"role": role, "content": content,
                             "session_date": session_date,
                             "source": source})
            if len(msgs) >= max_msgs:
                break
        if not msgs:
            msgs = [{"role": "user", "content": memory,
                     "session_date": session_date,
                     "source": source}]
        return msgs

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """中英文混合分词：英文词 + 中文 2-gram。"""
        import re
        text = text.lower()
        words = set(re.findall(r"[a-z0-9]+", text))
        han = re.findall(r"[一-鿿]+", text)
        for h in han:
            for i in range(max(0, len(h) - 1)):
                words.add(h[i:i+2])
        return words

    @classmethod
    def _overlap(cls, q: Set[str], text: str) -> float:
        """词重叠度（占位相似度，explore agent 阶段换 embedding）。"""
        words = cls._tokenize(text)
        if not q or not words:
            return 0.0
        return len(q & words) / len(q | words)


# 便捷：单例（服务复用）
_store = WikiStore()


def get_store() -> WikiStore:
    return _store
