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
import math
import re
from collections import Counter
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
    role: str = ""                     # user/assistant（count-hint 区分用户自述 vs 助手推荐）
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


# ---------------- BM25 检索 ----------------

# 英文停用词（检索噪音）：query 和文档词项统一过滤，让分数集中在内容词上。
# 注意：否定词 not/never 故意不进停用词——虽会增加少量噪音，
# 但 personamem 偏好更新依赖极性判断（_is_negation + polarity 头），
# 召回阶段保留 not 可避免"doesn't like"被误召回为"likes"。
BM25_STOP = set("""a an the and or but if because when what which who whom this that these those
i me my we our you your he his him she her they them their it its is are was were be been being
am do does did doing have has had having to of in on at for from by with without as no so
than then too very just about some any all can could would should will shall may might must
there here where how why up down out off over under again further once above below between
please could tell know need want going get got""".split())

_BM25_RE_HAN = re.compile(r"[一-鿿]+")
_BM25_RE_WORD = re.compile(r"[a-z0-9]+")


def _singular_form(word: str) -> str:
    """保守英文单数化（规则复数 → 单数，供 BM25 词形归一）。

    full_input(500 条)下最大召回杀手是 query 复数 vs 消息单数：
    "tanks"匹配不到"tank"、"festivals"匹配不到"festival" → BM25 分数 0，
    答案消息排不进 top-k。这里把英文词规约到单数（tank/tanks → tank），
    索引与 query 两侧同时归一，单复数即等价。

    只处理规则复数，保护以 -ss/-us/-is/-as/-os 结尾的固有单数（bus/class/
    this/analysis/gas 等），避免过度剥离。
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"                  # babies -> baby, movies 不命中
    if word.endswith("sses"):
        return word[:-2]                        # classes -> class
    if word.endswith(("xes", "ches", "shes", "zes")) and len(word) > 4:
        return word[:-2]                        # boxes -> box, watches -> watch
    if word.endswith("ses") and len(word) > 4:
        return word[:-1]                        # houses -> house, cases -> case
    if word.endswith("es") and len(word) > 3:
        return word[:-2]                        # 其他 -es（goes -> go）
    if word.endswith("s") and not word.endswith(("ss", "us", "is", "as", "os")):
        return word[:-1]                        # tanks -> tank, hours -> hour
    return word


def _bm25_tokenize(text: str) -> List[str]:
    """中英文混合词项（带频率）：英文词（去停用词+单数化）+ 中文 2-gram。

    与 _tokenize（集合，词重叠度用）不同：保留频率，供 BM25 计算词频饱和。
    英文词统一规约到单数（full_input 长对话下复数 query 匹配单数消息的命门）。
    """
    text = text.lower()
    terms: List[str] = []
    for w in _BM25_RE_WORD.findall(text):
        if w not in BM25_STOP:
            terms.append(_singular_form(w))
    for h in _BM25_RE_HAN.findall(text):
        for i in range(max(0, len(h) - 1)):
            terms.append(h[i:i + 2])
    return terms


class BM25Index:
    """真实 BM25：IDF + 词频饱和 k1 + 长度归一 b（Lucene 变体 IDF 公式）。

    不可变索引：写入后重建（WikiStore 按 user 缓存，ingest 时失效）。
    docs 是 section.content 列表，与调用方 section 顺序一一对应。
    """

    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.docs = list(docs)
        self.k1 = k1
        self.b = b
        self.doc_terms = [_bm25_tokenize(d) for d in self.docs]
        self.dl = [len(t) for t in self.doc_terms]
        self.N = len(self.docs)
        self.avgdl = sum(self.dl) / max(self.N, 1)
        # 文档频率（df）：词出现在多少篇文档里（每篇只计一次）
        self.df: Dict[str, int] = {}
        for terms in self.doc_terms:
            for t in set(terms):
                self.df[t] = self.df.get(t, 0) + 1
        # IDF（Lucene 变体，避免除零；新词（df=0）idf 取 0，不贡献分数）
        self.idf = {t: math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))
                    for t, n in self.df.items()}
        self._tfs = [Counter(t) for t in self.doc_terms]
        # 长度归一常数预计算：k1 * (1 - b + b * dl/avgdl)
        self._denoms = [self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                        for dl in self.dl]

    def score_doc(self, qterms: List[str], di: int) -> float:
        tf = self._tfs[di]
        denom = self._denoms[di]
        s = 0.0
        for t in qterms:
            f = tf.get(t, 0)
            if f:
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + denom)
        return s

    def scores(self, qterms: List[str]) -> List[float]:
        return [self.score_doc(qterms, i) for i in range(self.N)]


# ---------------- 存储层 ----------------

class WikiStore:
    """按 user_id 隔离的 wiki 记忆库。"""

    LINK_TYPES = ("related_to", "temporal_next", "caused_by", "contrasts_with")

    def __init__(self):
        # user_id -> {dim_id -> Dimension}
        self.users: Dict[str, Dict[str, Dimension]] = {}
        # user_id -> 用户级元数据（如 Current Date，时序题 "X days ago" 参考锚点）
        self.user_meta: Dict[str, Dict] = {}
        # 线程锁：FastAPI 多线程并发 Add/Search，写操作需要保护
        self._lock = __import__("threading").RLock()
        # BM25 索引缓存：user_id -> (write_ver, BM25Index)。ingest 后失效重建。
        self._bm25_cache: Dict[str, tuple] = {}
        self._bm25_ver: Dict[str, int] = {}  # user_id -> 写入版本
        # 检索融合配置（RRF 参数可调，评测对比用）：
        #   rrf_k       RRF 常数 k（越小越看重靠前 rank）
        #   w_kw/w_emb  关键词/语义两腿权重（=1 即标准 RRF 等权）
        #   score_mode  返回证据的 score 字段："rrf" 用融合分（推荐，驱动顶层排序），
        #               "emb" 用 embedding 余弦（旧行为），"kw" 用 BM25 分
        self.search_cfg: Dict = {"rrf_k": 60, "w_kw": 1.0, "w_emb": 1.0, "score_mode": "rrf"}

    def set_user_meta(self, user_id: str, key: str, value) -> None:
        with self._lock:
            self.user_meta.setdefault(user_id, {})[key] = value

    def get_user_meta(self, user_id: str, key: str, default=None):
        return self.user_meta.get(user_id, {}).get(key, default)

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
        # 写入会改变 section 集合 → BM25 索引失效（懒重建）
        self._bm25_ver[user_id] = self._bm25_ver.get(user_id, 0) + 1
        self._bm25_cache.pop(user_id, None)
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

        # 提取 "Current Date"（时序题 "X days ago" 参考锚点，Super Bowl 题证明是命门）
        # v2.5：parse_dialog 已把 focused 尾部 "Current Date" 附到每条消息的
        # msg["current_date"]（旧代码只扫 content，而 "Current Date:" 不在任何消息
        # content 里 → user_meta 永远是 None，as-of 永远上送不了）。
        current_date = None
        try:
            from time_utils import extract_current_date
            for msg in messages:
                current_date = msg.get("current_date") or extract_current_date(msg.get("content", ""))
                if current_date:
                    break
            if current_date is not None:
                self.set_user_meta(user_id, "current_date", str(current_date))
        except Exception:
            pass

        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", "")
            if not content:
                continue

            # source 兜底：_expand_neighbors 按 source 分组判相邻，空 source 会全失效
            if not msg.get("source"):
                msg["source"] = session_id or f"{user_id}:batch"

            # 相对时间归一化（v2.5：锚点用**本条消息自己的会话日期**——"today/yesterday/
            # X days ago" 相对它所在会话算。旧代码拿全批第一个会话日期当全局锚，
            # 多会话 focused 里第二/第三会话的 "today" 全被压平到第一个会话日期，
            # "between events" 题系统性算成 0 天。current_date 只做 as-of 锚点，不做归一化锚）
            try:
                from time_utils import normalize_relative_times
                ref = msg.get("session_date") or session_date or current_date
                content = normalize_relative_times(content, ref_date=ref)
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
                            role=msg.get("role", ""),      # user/assistant
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
                        role=msg.get("role", ""),      # user/assistant
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
    def _collect_sections(self, user_id: str) -> List[tuple]:
        """收集 (dim, page, section)，保证与 BM25 索引的 docs 顺序一致。"""
        dims = self._ensure_user(user_id)
        sections: List[tuple] = []
        for dim in dims.values():
            for page in dim.pages.values():
                for section in page.sections.values():
                    sections.append((dim, page, section))
        return sections

    def _get_bm25(self, user_id: str, sections: List[tuple]) -> BM25Index:
        """按 user 取 BM25 索引（带写入版本缓存）。"""
        docs = [sec.content for _, _, sec in sections]
        ver = self._bm25_ver.get(user_id, 0)
        cached = self._bm25_cache.get(user_id)
        if cached and cached[0] == ver:
            return cached[1]
        index = BM25Index(docs)
        self._bm25_cache[user_id] = (ver, index)
        return index

    def keyword_search(self, user_id: str, query: str, top_k: int = 10) -> List[Dict]:
        """关键词检索：真实 BM25（IDF + 词频饱和 + 长度归一），替换旧词重叠度。"""
        sections = self._collect_sections(user_id)
        if not sections:
            return []
        index = self._get_bm25(user_id, sections)
        qterms = _bm25_tokenize(query)
        scores = index.scores(qterms)
        ranked = sorted(range(len(sections)), key=lambda i: -scores[i])
        results = []
        for i in ranked:
            if scores[i] <= 0:
                break  # 已按分数降序，剩余都 <=0
            d, p, sec = sections[i]
            results.append({
                "id": p.id,
                "content": sec.content,
                "score": round(scores[i], 4),
                "page_title": p.title,
                "dimension": d.name,
                "created_at": sec.temporal,
                "role": sec.role,
                "source": sec.source,
            })
            if len(results) >= top_k:
                break
        return results

    def hybrid_search(self, user_id: str, query: str, top_k: int = 10) -> List[Dict]:
        """
        混合检索：BM25 词频 + 语义向量，RRF 融合排序（借鉴 Mem0 / Engram）。
        补词频抓不住的语义关联（如 query "items of clothing" ↔ "pick up dry cleaning"）。
        embedding 不可用时降级纯词频。
        RRF 参数（k、两腿权重）和 score 模式由 self.search_cfg 控制（评测调优用）。
        """
        cfg = self.search_cfg
        K = float(cfg.get("rrf_k", 60))
        w_kw = float(cfg.get("w_kw", 1.0))
        w_emb = float(cfg.get("w_emb", 1.0))
        score_mode = cfg.get("score_mode", "rrf")

        # 1. 收集全部 section，docs 顺序 = mem_map 键顺序（保证 rank 对齐）
        sections = self._collect_sections(user_id)
        if not sections:
            return []
        docs = [sec.content for _, _, sec in sections]
        mem_map = {}
        for d, p, sec in sections:
            mem_map[sec.content] = {
                "id": p.id,
                "content": sec.content,
                "score": 0.0,
                "page_title": p.title,
                "dimension": d.name,
                "source": sec.source,       # 证据溯源
                "role": sec.role,           # user/assistant(count-hint 过滤用)
                "temporal": sec.temporal,   # 时间锚点
                "confidence": sec.confidence,  # 抽取置信度
                "polarity": sec.polarity,   # 极性（Fable5：否定≠低置信）
                "order": sec.order,         # 消息序号（滑动窗口用）
                "speaker": sec.facts[0] if sec.facts else "",  # speaker 占位
            }

        # 2. BM25 全量 rank（不只 top_k，让 RRF 能看到全部命中文档）
        index = self._get_bm25(user_id, sections)
        qterms = _bm25_tokenize(query)
        kw_scores = index.scores(qterms)
        kw_score_map = {docs[i]: kw_scores[i] for i in range(len(docs))}
        kw_rank = {}
        order = sorted(range(len(docs)), key=lambda i: -kw_scores[i])
        for rank, i in enumerate(order):
            if kw_scores[i] <= 0:
                break
            kw_rank[docs[i]] = rank + 1

        # 3. 语义向量 rank（use_emb=False 时跳过——full_input 500条长对话 embedding
        #    极慢(~90s/条),且长消息截断后语义不可靠;评测迭代和定位时用 BM25 单腿快跑）
        emb_rank = {}
        emb_sim = {}
        use_emb = bool(cfg.get("use_emb", True))
        if docs and use_emb:
            from embedder import get_embedder
            sims = get_embedder().search(query, docs, top_k=top_k)
            # 全部相似度为 0 = embedding 不可用/失败 → 跳过语义腿（避免 0 分噪音 rank）
            if max(sims) > 0:
                emb_sim = {docs[i]: sims[i] for i in range(len(docs))}
                eorder = sorted(range(len(docs)), key=lambda i: -sims[i])
                for rank, i in enumerate(eorder):
                    emb_rank[docs[i]] = rank + 1
                    mem_map[docs[i]]["score"] = round(float(sims[i]), 4)

        # 4. RRF 融合：score = w_kw/(K+kw_rank) + w_emb/(K+emb_rank)
        rrf = {}
        for c in mem_map:
            s = 0.0
            if c in kw_rank:
                s += w_kw / (K + kw_rank[c])
            if c in emb_rank:
                s += w_emb / (K + emb_rank[c])
            rrf[c] = s

        # 5. 按 RRF 排序，取 top_k；只保留至少命中一腿的（RRF>0）
        sorted_content = sorted(rrf, key=rrf.get, reverse=True)
        results = [mem_map[c] for c in sorted_content[:top_k]]
        results = [r for r in results if rrf.get(r["content"], 0) > 0]

        # 6. 证据 score（explore._dedup 按它排序 → 决定 answer 模型看到的 top 顺序）
        for r in results:
            c = r["content"]
            if score_mode == "rrf":
                r["score"] = round(rrf.get(c, 0.0), 6)
            elif score_mode == "kw":
                r["score"] = round(kw_score_map.get(c, 0.0), 4)
            # score_mode == "emb" → 保持 embedding 余弦（已在第 3 步设置）

        # 滑动窗口扩展：命中消息的相邻消息也补召回（对话连续性，locomo/scriptmem 事件簇）
        results = self._expand_neighbors(user_id, results, window=2, max_extra=10)
        # count 聚簇提示由 api.py 的 LLM 实体提取(build_count_hint)负责，这里不重复
        # 注意：近重复去重(_dedup_similar) 未验证出提升，暂不启用（避免未验证改动）
        # 原 _cluster_count_hint(词频聚簇) 已删除：死代码 + "theme:msg_count" 形态接近违规
        return results[:top_k]

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
        # v2.7：邻居消息分必须低于所有真实命中，否则 explore._dedup 按 score 降序
        # 重排时邻居挤到 top（full_input 500 条下 RRF 分 ~0.016 < 0.3），把答案消息
        # 挤出 top-10 → 时序/count 题 INSUFFICIENT。取命中最低分做地板，保证邻居殿后。
        floor = 0.0
        if results:
            try:
                floor = min(float(r.get("score", 0) or 0) for r in results) - 0.001
            except Exception:
                floor = 0.0
        floor = max(floor, 0.0)  # 不为负
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
                                "id": page_id, "content": content, "score": floor,
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

        v2.5 修复（时序题 "X days between events" 命门）：
        focused_input 可能含**多个会话块**（每个 "Session YYYY/MM/DD" 头后面跟若干消息），
        此前所有消息都拿第一个会话的日期当锚点 → 第二/第三会话里 "today" 归一化到
        第一个会话日期，事件日全被压平 → "between events" 题算成 0 天。
        现在每条消息归属其**最近的 Session 头**，用自己的会话日期做锚。
        同时把 focused 尾部的 "Current Date"（"X days ago" 参考锚点）附到每条消息，
        ingest 才能写进 user_meta。
        """
        import re
        import datetime as _dt
        # Current Date 在 focused_input 尾部（"Current Date: 2023/03/01 ..."），
        # 不在任何消息 content 里——直接整串扫，附到每条消息供 ingest 用
        current_date = None
        try:
            from time_utils import extract_current_date
            current_date = extract_current_date(memory)
        except Exception:
            pass

        # 收集所有 Session 头（含位置）。每条消息归属它前面最近的 Session。
        # 格式：History Chats:Session 2023/02/14 (Tue) 16:29: / Session 2023/03/15 (Wed) 08:37:
        headers = []  # (pos, date, source)
        for m in re.finditer(r"Session\s+(\d{4})/(\d{1,2})/(\d{1,2})[^\n]*", memory):
            try:
                d = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
            src = m.group(0).strip()
            ci = src.find(":")  # 到行内第一个冒号（去掉秒），保持旧 source 风格
            if ci != -1:
                src = src[:ci]
            headers.append((m.start(), d, src.strip()))

        def _nearest(pos: int):
            """pos 前最近的 session 头 (date, source)；无则回退 ('dialog')。"""
            date, src = None, "dialog"
            for hpos, hdate, hsrc in headers:
                if hpos < pos:
                    date, src = hdate, hsrc
                else:
                    break
            return date, src

        def _ts(d):
            if d is None:
                return None
            try:
                return int(_dt.datetime(d.year, d.month, d.day,
                                        tzinfo=_dt.timezone.utc).timestamp() * 1000)
            except Exception:
                return None

        msgs = []
        # content 用双引号或单引号包裹（如 "I'm making progress..." / 'content'）。
        # 兼容两种引号，避免单引号 content 丢消息。
        pattern = (r"\{'role':\s*'(\w+)',\s*'content':\s*"
                   r"(?:\"((?:[^\"\\]|\\.)*)\"|'((?:[^'\\]|\\.)*)')}")
        for m in re.finditer(pattern, memory):
            role, content = m.group(1), m.group(2) if m.group(2) is not None else m.group(3)
            content = content.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
            if not content.strip():
                continue
            sdate, src = _nearest(m.start())
            msgs.append({"role": role, "content": content,
                         "session_date": sdate,
                         "timestamp": _ts(sdate),
                         "source": src,
                         "current_date": current_date})
            if len(msgs) >= max_msgs:
                break
        if not msgs:
            sdate, src = (headers[0][1], headers[0][2]) if headers else (None, "dialog")
            msgs = [{"role": "user", "content": memory,
                     "session_date": sdate,
                     "timestamp": _ts(sdate),
                     "source": src,
                     "current_date": current_date}]
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
    def _tokenize_terms(cls, text: str) -> List[str]:
        """带频率的中英文词项（去英文停用词）——BM25 用。"""
        return _bm25_tokenize(text)

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
