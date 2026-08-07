"""
MemFusion v2: 时序题计算提示生成（days ago / days passed between 命门）

answer 模型(qwen-plus 模拟平台)日期算术很弱：即使证据带 [date:] 元数据头，
"days ago" 会算偏、"days passed between" 直接输出 0（两个事件日期一样时的产物）。
这里用证据里确定性的 [date:] 元数据做算术，把结果以 [time-hint] 注入证据，
answer 模型只须照抄，不依赖它做日期减法。

只在日期充分时才触发（保守：算不准宁可不提示，避免污染召回）。
日期来源：证据的 temporal 字段（epoch ms，parse_dialog 按 Session 头分配，写入侧保证）。
"""
from __future__ import annotations

import datetime
from typing import List, Dict, Optional


def _collect_dates(results: List[Dict]) -> List[datetime.date]:
    """从证据里收集确定性的事件日期（去重、排序）。as-of/count-hint 条目无 temporal，自动排除。"""
    dates = []
    for r in results:
        ts = r.get("temporal")
        if not ts:
            continue
        try:
            d = datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc).date()
            dates.append(d)
        except Exception:
            pass
    return sorted(set(dates))


def parse_date(s: Optional[str]) -> Optional[datetime.date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(s.strip(), fmt).date()
        except Exception:
            pass
    return None


_BETWEEN_MARKERS = (
    "days passed between", "days had passed since", "days did it take",
    "passed between", "between the day", "how many days between",
    "days between", "weeks between", "months between",
)


def build_temporal_hint(query: str, results: List[Dict],
                        asof_str: Optional[str] = None) -> Optional[str]:
    """检测时序题；日期足够时返回 [time-hint] 证据条目内容，否则 None。"""
    ql = query.lower()
    is_ago = " ago" in ql or ql.endswith("ago")
    is_between = any(p in ql for p in _BETWEEN_MARKERS)
    if not (is_ago or is_between):
        return None

    dates = _collect_dates(results)
    if not dates:
        return None

    if is_ago:
        if not asof_str:
            return None
        asof = parse_date(asof_str)
        if not asof:
            return None
        ev = dates[-1]  # 最近的事件日期（该事件发生在距今最近的那条记忆）
        delta = (asof - ev).days
        if "week" in ql:
            val, unit = delta // 7, "weeks"
        elif "month" in ql:
            val, unit = delta // 30, "months"
        elif "hour" in ql or "minute" in ql:
            return None  # 小时/分钟级不在日期层面处理，避免算错
        else:
            val, unit = delta, "days"
        return (f"[time-hint] The current date (today) is {asof}. "
                f"The relevant event happened on {ev}, which is {val} {unit} before today. "
                f"So the answer is {val}.")

    if is_between:
        if len(dates) < 2:
            return None
        d0, d1 = dates[0], dates[-1]
        delta = (d1 - d0).days
        return (f"[time-hint] The two relevant event dates are {d0} and {d1}. "
                f"The number of days between them is {delta}. "
                f"So the answer is {delta} days.")

    return None
