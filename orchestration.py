"""
MemFusion v2：编排轨迹层（借鉴 arXiv:2605.02801）

论文核心：RL 要优化 agent 的编排决策（spawn/委托/通信/聚合/停止），
其中"停止决策"截至 2026-05 没有显式 RL 方法 —— 空白。

我们把 explore agent 的每次探索记录成编排轨迹（orchestration trace），
并显式建模 5 个子决策。v2.1 升级：轨迹可训练化。

5 个子决策（对应论文）：
- spawn:     是否/何时派子探索（我们的 explore agent 本身就是"子探索"）
- delegate:  委派给哪个工具/路径
- communicate: 如何组合多路结果
- aggregate: 如何聚合证据
- stop:      何时停止（证据够了）—— 论文点名的空白

可训练化设计（v2.1）：
1. 每条轨迹带 reward 标签（这次探索有没有找到答案）→ 可做 RL 训练数据
2. 停止决策可插拔（启发式 / 可学策略，相同接口）
3. 轨迹可导出为训练样本（JSON）
"""
from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class TraceEvent:
    """编排轨迹的单个事件。"""
    step: int
    decision: str          # spawn / delegate / communicate / aggregate / stop
    action: str            # 具体动作描述
    result: Any = None     # 该步的结果
    meta: Dict = field(default_factory=dict)


class StopPolicy:
    """
    停止策略接口（可插拔，对齐论文"停止决策可 RL 训练"）。
    子类实现 should_stop；当前提供启发式，将来可接 RL 训练的策略。
    """
    def should_stop(self, evidence: List[Dict], step: int, max_steps: int) -> bool:
        raise NotImplementedError


class HeuristicStop(StopPolicy):
    """启发式停止：evidence 最高置信度 >= threshold，或步数耗尽。"""
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def should_stop(self, evidence: List[Dict], step: int, max_steps: int) -> bool:
        if step >= max_steps:
            return True
        if evidence:
            top = max(e.get("score", 0) for e in evidence)
            if top >= self.threshold:
                return True
        return False


class Orchestrator:
    """
    编排器：记录 explore agent 的编排轨迹，管理 5 个子决策。
    v2.1：停止策略可插拔（默认启发式），轨迹带 reward 标签，可导出训练数据。
    """

    def __init__(self, max_steps: int = 5, stop_policy: Optional[StopPolicy] = None):
        self.max_steps = max_steps
        self.stop_policy = stop_policy or HeuristicStop(threshold=0.1)
        self.trace: List[TraceEvent] = []
        self.reward: Optional[float] = None  # 本条轨迹的 reward（训练用）

    def start_trace(self, query: str) -> List[TraceEvent]:
        self.trace = [TraceEvent(0, "spawn", f"start explore: {query}")]
        self.reward = None
        return self.trace

    def log(self, decision: str, action: str, result: Any = None, step: int = 0) -> None:
        self.trace.append(TraceEvent(step, decision, action, result))

    def should_stop(self, evidence: List[Dict], step: int) -> bool:
        """停止决策：委托给 stop_policy（启发式或可学策略）。"""
        return self.stop_policy.should_stop(evidence, step, self.max_steps)

    def set_reward(self, reward: float) -> None:
        """
        设置本条轨迹的 reward（训练用）。
        由外部在评测后调用：找到正确答案 → 正 reward；否则 → 负 reward。
        """
        self.reward = reward

    def get_trace(self) -> List[Dict]:
        return [{
            "step": e.step,
            "decision": e.decision,
            "action": e.action,
            "result": e.result,
        } for e in self.trace]

    def export_training_sample(self) -> Dict:
        """
        导出为 RL 训练样本：
        - 状态序列（各步决策 + 结果）
        - reward 标签
        - 最终 evidence（如果有）
        """
        return {
            "trace": self.get_trace(),
            "reward": self.reward,
            "max_steps": self.max_steps,
            "timestamp": time.time(),
        }

    def export_traces_to_jsonl(self, path: str) -> None:
        """把所有已记录的轨迹导出为 JSONL（RL 训练数据）。"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.export_training_sample(), ensure_ascii=False) + "\n")


def make_orchestrator(max_steps: int = 5, stop_threshold: float = 0.1) -> Orchestrator:
    return Orchestrator(max_steps=max_steps, stop_policy=HeuristicStop(threshold=stop_threshold))
