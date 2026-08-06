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

EXTRACT_PROMPT = """你是记忆助手。根据问题,从下面的对话消息里提取与答案相关的**具体实体**列表。

问题: {question}
对话消息:
{messages}

请只输出 JSON(不要其他内容):
{{"entities": ["实体1", "实体2", ...]}}

entities 规则:
- 是**具体事物**(衣物名、医生名、水果名、活动名等),不是抽象词
- 去重(同一实体只列一次)
- 只列与问题答案相关的
"""

# 词法降级：提取 Dr.X / 数字+单位 类模式
_PATTERNS = [
    r"Dr\.\s*[A-Z][a-z]+",
    r"Dr\s+[A-Z][a-z]+",
    r"\d+\s*(?:days?|weeks?|hours?|months?|years?)",
]


def is_count_query(query: str) -> bool:
    ql = query.lower()
    return any(w in ql for w in ["how many", "how much", "count", "number of",
                                 "几", "多少", "几个"])


def _lexical_entities(contents: List[str]) -> List[str]:
    """词法降级：Dr.X / 数字单位 类实体。"""
    joined = " ".join(contents)
    ents = set()
    for p in _PATTERNS:
        ents.update(re.findall(p, joined))
    return sorted(ents)


def extract_entities(query: str, contents: List[str],
                     api_key: str = "", model: str = "gpt-5.4-mini",
                     base_url: str = DEFAULT_URL,
                     timeout: int = 25) -> Optional[List[str]]:
    """LLM 提取实体。无 key 或失败返回 None（调用方降级词法）。"""
    if not api_key or not contents:
        return None
    msgs_text = "\n".join("· " + c[:150] for c in contents[:15])
    prompt = EXTRACT_PROMPT.format(question=query, messages=msgs_text)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
    }).encode()
    req = urllib.request.Request(base_url, data=body,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        out = d["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning("entity extract LLM failed: %s", e)
        return None
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        return None
    try:
        ents = json.loads(m.group(0)).get("entities", [])
    except Exception:
        return None
    return [str(e).strip() for e in ents if str(e).strip()]


def build_count_hint(query: str, results: List[Dict],
                     api_key: str = "", model: str = "gpt-5.4-mini",
                     base_url: str = DEFAULT_URL) -> List[Dict]:
    """
    对 count query 的召回结果，生成实体聚簇证据并追加。
    LLM 优先，失败降级词法。返回追加 [entities] 后的 results。
    """
    if not is_count_query(query) or not results:
        return results
    contents = [r.get("content", "") for r in results if r.get("content")]
    if not contents:
        return results

    # 已有词法 count-hint 先去掉（避免重复）
    results = [r for r in results if r.get("id") != "count-hint"]

    entities = extract_entities(query, contents, api_key=api_key,
                                model=model, base_url=base_url)
    if not entities:
        entities = _lexical_entities(contents)
    if not entities:
        return results

    hint = "[count-hint] entities: " + ", ".join(entities)
    results.append({
        "id": "count-hint", "content": hint, "score": 0.5,
        "page_title": "", "dimension": "", "source": "", "order": -1,
    })
    return results
