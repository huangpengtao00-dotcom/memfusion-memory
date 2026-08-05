"""
MemFusion v2：相对时间归一化（Fable5 建议，locomo 数据集关键）

把记忆里的相对时间（"昨天/上周/几天前"）转成绝对日期，
让 answer 模型能正确做时间推理（locomo 测"昨天→日期"）。
"""
from __future__ import annotations

import re
import datetime
from typing import Optional


def normalize_relative_times(text: str, ref_date: Optional[datetime.date] = None) -> str:
    """
    把 text 里的相对时间词替换为绝对日期（基于 ref_date）。
    ref_date = 对话发生的日期（默认今天）。
    示例：ref=2023/02/15 时，"yesterday" → "2023-02-14"。
    """
    ref = ref_date or datetime.date.today()

    def replace(m):
        w = m.group(0).lower()
        delta = {
            "today": 0, "yesterday": -1, "tomorrow": 1,
            "前天": -2, "昨天": -1, "今天": 0, "明天": 1, "后天": 2,
        }.get(w)
        if delta is not None:
            return str(ref + datetime.timedelta(days=delta))
        return w

    # 中文相对时间
    for kw in ["前天", "昨天", "今天", "明天", "后天"]:
        text = re.sub(kw, replace, text)
    # 英文相对时间
    for kw in ["today", "yesterday", "tomorrow"]:
        text = re.sub(rf"\b{kw}\b", replace, text, flags=re.I)
    # last/this/next + 星期（英文 + 中文）
    for pattern in [r"(?i)\b(last|this|next)\s+(mon|tue|wed|thu|fri|sat|sun)(day)?\b",
                    r"(上|这|下)(周|星期)(一|二|三|四|五|六|日|天)"]:
        text = re.sub(pattern, lambda m: _weekday_replace(m, ref), text)
    return text


def _weekday_replace(m, ref: datetime.date) -> str:
    """上周/这周/下周 + 星期X → 绝对日期。"""
    direction = m.group(1)
    weekday_map = {
        "一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
        "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
    }
    # 兼容中文 (上|这|下)(周)(五) 的 weekday 在 group(3)，英文 (last)(fri)(day) 在 group(2)
    wd = None
    if m.group(3) and m.group(3).lower() in weekday_map:
        wd = m.group(3)
    elif m.group(2) and m.group(2).lower() in weekday_map:
        wd = m.group(2)
    target = weekday_map.get(wd.lower() if wd else "")
    if target is None:
        return m.group(0)
    offset = {
        "上": -7, "last": -7, "这": 0, "this": 0, "下": 7, "next": 7,
    }.get(direction, 0)
    # 本周目标星期几
    this_wk = ref - datetime.timedelta(days=ref.weekday())  # 本周一
    d = this_wk + datetime.timedelta(days=target + offset)
    return str(d)


def extract_session_date(memory: str) -> Optional[datetime.date]:
    """从 LongMemEval full_input 提取对话日期（Session YYYY/MM/DD）。"""
    m = re.search(r"Session\s+(\d{4})/(\d{1,2})/(\d{1,2})", memory)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None
