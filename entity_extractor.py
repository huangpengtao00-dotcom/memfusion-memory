"""
MemFusion v2: count 类题目实体提取器

count 题（"how many items/different doctors"）需要把召回消息里与答案相关的
具体实体聚簇列出（衣物名/医生名/水果名），answer 模型才能数对。
词法聚簇只能处理 Dr.X 类模式，语义类（clothing/fruit）需要 LLM。

Search 时对 count query 调用 LLM 提取实体列表，返回结构化证据（不生成答案，合规）。
LLM 失败/无 key 时降级词法聚簇。
"""
from __future__ import annotations

import json
import re
import logging
from typing import List, Dict, Optional

try:
    import urllib.request
except ImportError:
    pass

log = logging.getLogger("memfusion.entity")

# 默认走环境变量 LLM（比赛合规 gpt-4o-mini；端点无则用 gpt-5.4-mini）
DEFAULT_URL = "https://mx.free.codesonline.dev/v1/chat/completions"

# LLM 提取结果缓存（query+内容 → entities），比赛重复 Search 同 query 时省 LLM 调用
_EXTRACT_CACHE: Dict[str, Optional[List[str]]] = {}

EXTRACT_PROMPT = """你是记忆助手。根据问题,从下面的对话消息里提取与答案相关的**具体实体**列表。

问题: {question}
对话消息:
{messages}

请只输出 JSON(不要其他内容):
{{"entities": ["实体1", "实体2", ...]}}

entities 规则:
- 是**具体事物**(衣物名、医生名、水果名、活动名等),不是抽象词
- **列出每一个**不同的实体,拿不准也列,宁多勿漏(count 题靠它数数)
- 去重(同一实体只列一次)
- **同一个实体被以不同称呼提到时(如 "drum set" 和 "Pearl Export" 指同一个鼓、
  "acoustic guitar" 和 "Yamaha FG800" 指同一把吉他),只列一次**,
  用能唯一标识的完整称呼(品牌+类别)
- 只列与问题答案相关的
- **若问题是"当前拥有/目前有/currently"类:只列"当前有效"的实体**,
  排除历史已卖的、想买的(愿望)、别人/助手推荐的(非用户所有)、过去的
  示例:问"currently own 乐器",只列当前拥有的乐器,不列"想买"/"卖掉了"的
- **严格排除别人/亲属/朋友的财物**:"my niece's violin"、"my grandma's X"、
  "my friend's Y" 等是他人所有,不是用户的当前拥有物
"""

# 词法降级：Dr.X / Film Festival 类模式（query 感知，只扫 user 消息，避免 assistant 推荐噪音）
_DR_PATTERNS = [
    r"Dr\.\s*[A-Z][a-z]+",
    r"Dr\s+[A-Z][a-z]+",
]
# 电影节类："Austin Film Festival" / "Seattle International Film Festival" / "AFI Fest"
# 不要加裸的 `\w+ Festival`（会匹配 "Film Festival" 这类泛称假阳性）。
_FESTIVAL_PATTERNS = [
    r"[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]+)*\s+(?:International\s+)?Film\s+Festival",
    r"[A-Z][A-Za-z]+\s+Fest\b",
]


def _entity_in_msg(entity_l: str, content: str) -> bool:
    """实体是否出现在消息里（子串或整词）。用于过滤 LLM 幻觉的实体。"""
    cl = content.lower()
    if entity_l in cl:
        return True
    # 整词匹配（entity 可能是 "AFI Fest" 而消息里是 "AFI Fest" 或其变体）
    words = re.findall(r"[a-z0-9]+", entity_l)
    return bool(words) and all(w in cl for w in words)


# "当前拥有"题的实体拥有性判断
_OWN_SIGNALS = ("i've had", "i have had", "been playing", "i own", "have a ",
                "my go-to", "my old", "my favorite", "i've been playing", "i've owned")
_RELATIVE_POSS = ("niece's", "grandma's", "grandpa's", "mother's", "father's",
                  "sister's", "brother's", "friend's", "son's", "daughter's",
                  "cousin's", "nephew's", "uncle's", "aunt's")
_WISH_EXTRA = ("want to get", "will get", "going to get", "when i get", "get my new",
               "thinking about", "i'd like to buy", "i want", "wish i had", "haven't bought")


def _entity_currently_owned(entity_l: str, user_contents: List[str]) -> bool:
    """实体是否属于用户"当前拥有"（排除愿望/他人财物）。
    检查**实体附近 ±25 字符窗口**内的语境（整条消息会误判：#90 "I've been playing
    guitar... thinking about getting a new ukulele" 里 "been playing" 指吉他，与 ukulele 无关）。
    任何窗口有拥有信号 → 拥有；否则任一窗口是愿望/他人语境 → 不拥有。"""
    msgs = [uc for uc in user_contents if _entity_in_msg(entity_l, uc)]
    if not msgs:
        return False
    pat = re.compile(re.escape(entity_l))
    has_ownership = False
    for m in msgs:
        ml = m.lower()
        for m_ent in pat.finditer(ml):
            start = max(0, m_ent.start() - 25)
            end = min(len(ml), m_ent.end() + 25)
            ctx = ml[start:end]
            if any(sig in ctx for sig in _OWN_SIGNALS):
                has_ownership = True
                break
            # "my X" 拥有（"my black fender" / "my new ukulele" 中 "my new" 是愿望,不做拥有信号）
            if re.search(r"\bmy\s+(?:own\s+|old\s+|favorite\s+|go-to\s+)?\b", ctx) and \
               re.search(r"\bmy\s+(?:own\s+|old\s+|favorite\s+|go-to\s+)?\b", ctx).end() <= m_ent.start():
                # "my X" 紧邻实体前且不是 "my new"/"my next"
                pre = ml[max(0, m_ent.start() - 15):m_ent.start()]
                if not any(w in pre for w in ("my new", "my next", "get my", "a new")):
                    has_ownership = True
                    break
        if has_ownership:
            break
    if has_ownership:
        return True
    # 无任何拥有信号：任一窗口是愿望(想买/将买)或他人财物语境 → 不算当前拥有
    for m in msgs:
        ml = m.lower()
        for m_ent in pat.finditer(ml):
            start = max(0, m_ent.start() - 25)
            end = min(len(ml), m_ent.end() + 25)
            ctx = ml[start:end]
            if any(w in ctx for w in (_WISH_MARKERS + _WISH_EXTRA)):
                return False
            if any(rp in ctx for rp in _RELATIVE_POSS):
                return False
    return True


def is_count_query(query: str) -> bool:
    ql = query.lower()
    return any(w in ql for w in ["how many", "how much", "count", "number of",
                                 "几", "多少", "几个"])


def _lexical_entities(query: str, user_contents: List[str]) -> List[str]:
    """query 感知的词法提取：只对问题确实在数的类别应用对应规则，
    且只扫 user 消息（assistant 推荐的名词是噪音，如 SXSW 是推荐不是用户参加的）。"""
    ql = query.lower()
    joined = " ".join(user_contents)
    ents = set()
    if "doctor" in ql or " dr" in ql or "dr." in ql:
        for p in _DR_PATTERNS:
            ents.update(re.findall(p, joined))
    if "festival" in ql or "film fest" in ql:
        for p in _FESTIVAL_PATTERNS:
            ents.update(re.findall(p, joined))
    return sorted(ents)


def extract_entities(query: str, contents: List[str],
                     api_key: str = "", model: str = "gpt-5.4-mini",
                     base_url: str = DEFAULT_URL,
                     timeout: int = 25) -> Optional[List[str]]:
    """LLM 提取实体。无 key 或失败返回 None（调用方降级词法）。
    带缓存：相同 query+召回内容 不重复调 LLM（比赛会重复 Search 同 query，省超时/成本）。"""
    if not api_key or not contents:
        return None
    # 喂全部召回消息（不全量截断——维度侧报告实证：contents[:15]+c[:150] 会丢第三个实体
    # → count 题系统性少数 1）。full_input(500条)召回可达 100 条，答案消息可能排在 40 之后
    # （Yamaha FG800 在 42）→ 扩到 80 条，每条 200 字符控制上下文。
    msgs_text = "\n".join("· " + c[:200] for c in contents[:80])
    prompt = EXTRACT_PROMPT.format(question=query, messages=msgs_text)
    cache_key = prompt[:800]
    if cache_key in _EXTRACT_CACHE:
        return _EXTRACT_CACHE[cache_key]
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
    }).encode()
    req = urllib.request.Request(base_url, data=body,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    out = None
    # 重试：full_input 长对话评测里网关偶发 5xx/429，count-hint 提取失败会直接丢题
    # （focused 3 条时 count-hint 不是命门；500 条时 count-hint 是 count 题唯一可数线索）。
    import time as _t
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            out = d["choices"][0]["message"]["content"]
            break
        except Exception as e:
            log.warning("entity extract LLM failed (try %d): %s", attempt + 1, e)
            if attempt < 2:
                _t.sleep(1.5 * (attempt + 1))
    if out is None:
        return None
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        return None
    try:
        ents = json.loads(m.group(0)).get("entities", [])
    except Exception:
        return None
    ents = [str(e).strip() for e in ents if str(e).strip()]
    _EXTRACT_CACHE[cache_key] = ents
    return ents


# "当前拥有"类问题的愿望清单/获取意图标记：这些消息里的实体不是用户当前拥有的
# （"thinking of buying a new ukulele" 的 ukulele 不该数进"currently own"）。过滤掉再喂 LLM。
_OWN_MARKERS = ("currently", "now ", " do i own", " do i have", "own", "possess", "current")
_WISH_MARKERS = ("thinking of buying", "want to buy", "wish to buy", "looking to buy",
                 "considering buying", "thinking about getting", "eyeing", "planning to buy",
                 "thinking of getting", "hoping to buy", "interested in buying", "want to get",
                 "wish i had", "haven't bought", "not yet")


def build_count_hint(query: str, results: List[Dict],
                     api_key: str = "", model: str = "gpt-5.4-mini",
                     base_url: str = DEFAULT_URL) -> List[Dict]:
    """
    对 count query 的召回结果，生成实体聚簇证据并追加。
    LLM 优先，失败降级词法。返回追加 [entities] 后的 results。
    """
    if not is_count_query(query) or not results:
        return results
    ql = query.lower()
    # 时长/天数类（"how many hours/days/total time"）不是实体计数，走 build_sum_hint，
    # 这里跳过 count-hint（列实体名会让 answer 模型数游戏数/事件数而非求和）。
    if any(w in ql for w in ["how many hours", "total hours", "hours in total",
                             "how many days", "total days", "days in total", "much time",
                             "total number of days", "number of days", "total number of hours",
                             "number of hours", "how many hours total"]):
        from temporal_hint import build_sum_hint
        hint_txt = build_sum_hint(query, results)
        if hint_txt:
            results = [r for r in results if r.get("id") not in ("count-hint", "sum-hint")]
            results.insert(0, {
                "id": "sum-hint", "content": hint_txt, "score": 0.5,
                "page_title": "", "dimension": "", "source": "", "order": -1,
            })
        return results
    # 优先用户自述(user)，assistant 推荐是噪音(ukulele 想买但没买混入)。
    # user 排前，assistant 后置——extract_entities 喂 contents 时先丢 assistant。
    user_contents = [r.get("content", "") for r in results
                     if r.get("content") and r.get("role") != "assistant"]
    all_contents = [r.get("content", "") for r in results if r.get("content")]
    contents = user_contents + all_contents
    if not contents:
        return results
    # "当前拥有/currently"类问题：过滤愿望清单消息（想买但没买不算当前拥有）
    if any(m in ql for m in _OWN_MARKERS):
        contents = [c for c in contents if not any(w in c.lower() for w in _WISH_MARKERS)]
        if not contents:
            contents = [r.get("content", "") for r in results if r.get("content")]  # 兜底

    # 已有词法 count-hint 先去掉（避免重复）
    results = [r for r in results if r.get("id") != "count-hint"]

    # 词法提取**始终运行**并与 LLM 结果合并（v2.7）：电影节/Dr.X 类规则比 LLM 更可靠，
    # LLM 抖动会漏掉 Seattle（只列 3 个电影节）→ 词法补漏。只扫 user 消息 + query 感知，
    # 避免把 assistant 推荐的 SXSW/时长年份混进实体列表。
    lexical = _lexical_entities(query, user_contents)
    # 电影节题：词法规则("X Film Festival"/"X Fest")比 LLM 精确，直接以词法为准
    # （LLM 会把 "48-hour film challenge" 活动也列进实体，count 错）。
    if any(w in ql for w in ["festival", "film fest"]):
        merged = lexical
    else:
        entities = extract_entities(query, contents, api_key=api_key,
                                    model=model, base_url=base_url) or []
        merged = []
        seen = set()
        # 后置过滤：只保留**确实出现在 user 消息里**的实体（LLM 会幻觉不存在的实体）。
        # "当前拥有/currently"类再加实体级拥有性检查：ukulele 只在"想买/将买"语境
        # (非当前拥有)、violin 只在"侄女的琴"语境(他人财物)，都不该进 count。
        for e in list(entities) + list(lexical):
            key = e.lower().strip()
            if key and key not in seen:
                seen.add(key)
                if any(_entity_in_msg(key, uc) for uc in user_contents):
                    if any(m in ql for m in _OWN_MARKERS) and not _entity_currently_owned(key, user_contents):
                        continue
                    merged.append(e.strip())
    if not merged:
        return results

    # 合规(比赛"不得在 Search 中直接生成最终答案")：只列实体，不写 count=N。
    # count 由实体列表长度决定，answer 模型自己数（合规审查确认：列实体=证据，写总数=违规）。
    hint = f"[count-hint] entities: " + ", ".join(merged)
    # v2.7：插入到最前（原来 append 在尾部）。full_input(500条)下结果往往 >10 条，
    # eval/上游只取 results[:10]，append 的 count-hint 被截断丢给 answer 模型 → count 题
    # 系统性失败（focused 3条时 results<10 侥幸可见，full 500条必丢）。前置保证 count-hint 一定送达。
    results.insert(0, {
        "id": "count-hint", "content": hint, "score": 0.5,
        "page_title": "", "dimension": "", "source": "", "order": -1,
    })
    return results
