# MemFusion v2 - AML code submission
# 平台构建部署用（代码提交方式）
FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastembed

# 代码
COPY . .

# 端口（AML 协议 Add/Search）
EXPOSE 8083

# 启动（key 从环境变量 MEMFUSION_LLM_API_KEY 注入）
CMD ["python3", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8083"]
