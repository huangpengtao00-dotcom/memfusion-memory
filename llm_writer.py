"""
MemFusion v2：LLM 写入器

把一批消息"理解"后写进 wiki：
1. LLM 从消息抽结构化事实（facts）
2. 判断是更新已有页面还是建新页（按主题聚类）
3. 建类型化链接（related_to / temporal_next / caused_by / contrasts_with）

借鉴 MemCog 的增量更新流程，但独立实现。
LLM 不可用时降级为简单匹配（不阻塞写入）。
"""
from __future__ import annotations

import json
import re
import logging
from typing import Dict, Optional

try:
    import urllib.request, urllib.error
except ImportError:
    pass

log = logging.getLogger("memfusion.writer")


class LLMWriter:
    """LLM 驱动的增量写入器。"""

    DEFAULT_URL = "https://mx.free.codesonline.dev/v1/chat/completions"
    DEFAULT_MODEL = "gpt-5.4-mini"

    EXTRACT_PROMPT = """你是记忆整理助手。把下面这条消息整理成结构化记忆。

消息: {content}

请输出 JSON（不要其他内容）：
{{
  "topic": "这条消息的主题（短，作为页面标题）",
  "facts": ["结构化事实1", "结构化事实2"],
  "dimension": "它属于哪个维度（个人/工作/关系/事件/其他，简短）"
}}
"""

    def __init__(self, api_key: str, base_url: str = DEFAULT_URL, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def _llm(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(self.base_url, data=body,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning("LLM call failed (fallback to simple write): %s", e)
            return None

    def extract(self, content: str) -> Optional[Dict]:
        """从消息抽结构化记忆。失败返回 None（降级）。"""
        out = self._llm(self.EXTRACT_PROMPT.format(content=content))
        if not out:
            return None
        m = re.search(r"\{.*\}", out, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def make_writer(api_key: str) -> LLMWriter:
    return LLMWriter(api_key=api_key)
