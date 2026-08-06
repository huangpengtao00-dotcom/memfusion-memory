"""
MemFusion v2：embedding 语义检索（借鉴 Mem0 / Engram 混合检索方案）

用 fastembed（bge-small-en）做语义向量检索，补词频抓不住的"语义关联"
（如 query "items of clothing" ↔ 记忆 "pick up dry cleaning navy blazer"）。
配合词频做 Hybrid RRF 融合。
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional


class Embedder:
    """轻量语义向量检索。惰性加载模型，失败时降级（不阻塞）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        self._dims = 0
        self._cache: Dict[str, list] = {}  # text -> embedding（缓存，评测同 query/记忆复用）

    def _ensure_model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
                # 取维度
                self._dims = len(next(self._model.embed(["warmup"])))
            except Exception:
                self._model = None  # 不可用 → 降级
        return self._model is not None

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """批量 embed（带缓存）。失败返回 None（降级到词频）。"""
        if not self._ensure_model():
            return None
        try:
            # 缓存命中（评测同 query/记忆重复调用多）
            uncached = [t for t in texts if t not in self._cache]
            if uncached:
                embs = list(self._model.embed(uncached))
                for t, e in zip(uncached, embs):
                    self._cache[t] = e
            return np.array([self._cache[t] for t in texts])
        except Exception:
            return None

    def search(self, query: str, memories: List[str], top_k: int = 10) -> List[float]:
        """
        语义相似度检索：返回 memories 每条的相似度分数。
        memories 为空或失败 → 返回全 0（词频兜底）。
        """
        if not memories:
            return []
        qv = self.embed([query])
        mv = self.embed(memories)
        if qv is None or mv is None:
            return [0.0] * len(memories)
        # 余弦相似度（防除零：norm 为 0 时置 0）
        q_norm = np.linalg.norm(qv[0])
        m_norm = np.linalg.norm(mv, axis=1, keepdims=True)
        qn = qv[0] / q_norm if q_norm > 1e-9 else np.zeros_like(qv[0])
        mn = np.where(m_norm > 1e-9, mv / np.maximum(m_norm, 1e-9), np.zeros_like(mv))
        # 除零/空向量会先产生 NaN 再被 nan_to_num 清理，提前补零避免 RuntimeWarning
        qn = np.nan_to_num(qn)
        mn = np.nan_to_num(mn)
        sims = (mn @ qn)
        # 清理 NaN/inf
        sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=0.0)
        return sims.tolist()


# 全局单例（服务复用）
_embedder = Embedder()


def get_embedder() -> Embedder:
    return _embedder
