#!/bin/bash
# MemFusion v2 启动脚本
cd "$(dirname "$0")"
exec python3 -m uvicorn api:app --host 0.0.0.0 --port 8083 "$@"
