"""MemFusion v2 LLM 配置

- 比赛要求 Add/Search 必须用 gpt-4o-mini
- key 必须从环境变量 MEMFUSION_LLM_API_KEY 读取（安全，不硬编码）
"""
import os

# 比赛合规模型（Add/Search 必须用 gpt-4o-mini）
LLM_BASE_URL = os.environ.get(
    "MEMFUSION_LLM_BASE_URL",
    "https://mx.free.codesonline.dev/v1",
)
# key 从环境变量读；未设置时留空（公开仓库安全）
LLM_API_KEY = os.environ.get("MEMFUSION_LLM_API_KEY", "")
FAST_MODEL = os.environ.get("MEMFUSION_FAST_MODEL", "gpt-4o-mini")
