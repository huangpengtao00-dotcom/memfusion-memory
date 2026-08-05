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
    def ingest(self, user_id: str, messages: List[Dict], writer=None) -> int:
        """
        把一批消息写入 wiki。
        writer 提供时：LLM 抽事实 → 按 topic 建页/更新 → 按 dimension 归维度。
        writer 为 None 时：降级为简单匹配（每条消息建/更新一页）。

        建链接逻辑：
        - 同一批消息抽出的 facts，如果 topic 不同但有重叠实体 → related_to
        - temporal 相邻的页面 → temporal_next（简化：同批页都算 temporal_next）
        """
        # 整批写入加锁：并发 Add 时保证原子性
        with self._lock:
            return self._ingest_unlocked(user_id, messages, writer)

    def _ingest_unlocked(self, user_id: str, messages: List[Dict], writer=None) -> int:
        """加锁内的实际写入逻辑。"""
        dims = self._ensure_user(user_id)
        written = 0
        pages_this_batch: List[Page] = []

        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue

            # 相对时间归一化（Fable5：locomo 时间推理关键）
            try:
                from time_utils import normalize_relative_times
                content = normalize_relative_times(content)
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
        return results

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
        msgs = []
        pattern = r"\{'role':\s*'(\w+)',\s*'content':\s*'((?:[^'\\]|\\.)*)'\}"
        for m in re.finditer(pattern, memory):
            role, content = m.group(1), m.group(2)
            content = content.replace("\\n", "\n").replace("\\'", "'")
            if content.strip():
                msgs.append({"role": role, "content": content})
            if len(msgs) >= max_msgs:
                break
        if not msgs:
            msgs = [{"role": "user", "content": memory}]
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
