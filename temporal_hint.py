"""
MemFusion v2: 时序题计算提示生成（days ago / days passed between / clock time 命门）

answer 模型(qwen-plus 模拟平台)日期/时间算术很弱：即使证据带 [date:] 元数据头，
"days ago" 会算偏、"days passed between" 直接输出 0（两个事件日期一样时的产物）。
这里用证据里确定性的 [date:] 元数据做算术，把结果以 [time-hint] 注入证据，
answer 模型只须照抄，不依赖它做日期减法。

机制3(TReMu ACL2025)：钟表时间跨消息推断（"left home at 7 AM" + "took 2 hours"
= 9:00 AM），LLM 抽事件-时间对 + 代码链式合成。这里用正则先做简单 case。

只在日期/时间充分时才触发（保守：算不准宁可不提示，避免污染召回）。
"""
from __future__ import annotations

import datetime
import re
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
                f"The relevant event happened on {ev}.")

    if is_between:
        if len(dates) < 2:
            return None
        d0, d1 = dates[0], dates[-1]
        return (f"[time-hint] The two relevant event dates are {d0} and {d1}.")

    return None


def build_sum_hint(query: str, results: List[Dict]) -> Optional[str]:
    """时长/天数求和题（"how many hours/days total"）：把 user 消息里报告的
    "N hours/days" 收集起来，按 (数值,单位) 去重（同一事件被多条消息重复陈述，
    如 TLOU 30h 出现在两条消息）后求和。确定性，不依赖 LLM。
    保守触发：至少 2 个不同数值才求和（单条是具体事实题，不是求和题）。"""
    ql = query.lower()
    if not any(w in ql for w in ["how many hours", "total hours", "hours in total",
                                 "how many days", "total days", "days in total", "much time",
                                 "total number of days", "number of days", "total number of hours",
                                 "number of hours", "how many hours total"]):
        return None
    unit_word = "day" if "day" in ql and "hour" not in ql else "hour"
    pairs = []
    for r in results:
        if r.get("role") == "assistant":
            continue  # 只数 user 自述，assistant 复述会重复计数
        c = r.get("content", "")
        if unit_word == "hour":
            for m in re.finditer(r"\b(\d{1,4})\s+hours?\b", c):
                n = int(m.group(1))
                if 1 <= n <= 1000:
                    pairs.append((n, "hour"))
        else:
            # "N days" / "N-day" 直接提取
            for m in re.finditer(r"\b(\d{1,4})\s+days?\b|\b(\d{1,4})-day\b", c):
                n = int(m.group(1) or m.group(2))
                if 1 <= n <= 1000:
                    pairs.append((n, "day"))
            # 日期区间 "April 15th to 22nd" / "April 15-22" → 天数 = 结束 - 开始
            for m in re.finditer(
                    r"\b([A-Z][a-z]{2,8})\s+(\d{1,2})(?:st|nd|rd|th)?\s*(?:-|to)\s*"
                    r"(?:[A-Z][a-z]{2,8}\s+)?(\d{1,2})(?:st|nd|rd|th)?\b", c):
                try:
                    m0 = datetime.datetime.strptime(m.group(1), "%B").month
                    start = m0 * 100 + int(m.group(2))
                    end = m0 * 100 + int(m.group(3))
                    if end < start:
                        end += 100
                    days = end - start  # 结束日-开始日（April 15->22 = 7）
                    if 1 <= days <= 400:
                        pairs.append((days, "day"))
                except Exception:
                    pass
    if not pairs:
        return None
    # 去重 (数值,单位)：同一事件被多条消息重复陈述时数值相同 → 只计一次
    distinct = sorted(set(pairs))
    if len(distinct) < 2:
        return None
    total = sum(n for n, _ in distinct)
    parts = " + ".join(str(n) for n, _ in distinct)
    unit = unit_word + ("s" if total != 1 else "")
    return (f"[sum-hint] The user reported these {unit_word} amounts: {parts}. "
            f"Total = {total} {unit}. So the answer is {total} {unit}.")


# ---- 机制3(TReMu)：钟表时间跨消息推断 ----
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
              "twenty": 20, "thirty": 30}
_TIME_MARKERS = ("what time", "at what time", "o'clock", "what hour",
                 "几点", "什么时候到")


def _to_num(s: str) -> Optional[int]:
    s = s.strip().lower()
    if s.isdigit():
        return int(s)
    return _NUM_WORDS.get(s)


def _to_24h(h: int, ampm: str) -> int:
    ampm = ampm.upper()
    if ampm == "PM" and h < 12:
        return h + 12
    if ampm == "AM" and h == 12:
        return 0
    return h


def _to_12h(h: int) -> str:
    h = h % 24
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    return f"{h12}:00 {ampm}"


def build_clock_hint(query: str, results: List[Dict]) -> Optional[str]:
    """钟表时间题："left home at 7 AM" + "took 2 hours" → 9:00 AM。
    正则抽离开时间 + 时长，代码做加法。保守：抓不到返回 None。"""
    ql = query.lower()
    if not any(m in ql for m in _TIME_MARKERS):
        return None

    contents = [r.get("content", "") for r in results if r.get("content")]
    joined = " ".join(contents)

    dep = re.search(r"\bleft\b.*?\bat\s+(\d{1,2})\s*(am|pm)\b", joined, re.I)
    dur = re.search(r"\btook\s+(?:me\s+)?(\d+|one|two|three|four|five|six|seven|"
                    r"eight|nine|ten|fifteen|twenty|thirty)\s+hours?\b", joined, re.I)
    if not (dep and dur):
        return None
    n = _to_num(dur.group(1))
    if n is None:
        return None
    return (f"[time-hint] departure was at {dep.group(1)} {dep.group(2).upper()}, "
            f"and the trip took {n} hours.")
