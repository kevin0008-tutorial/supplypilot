"""SupplyPilot 后端服务入口。

第1节：根路由 + 健康检查接口 /health。
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="SupplyPilot", version="0.1.0")


@app.get("/")
def root() -> dict:
    return {"service": "supplypilot", "docs": "/docs"}


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查：服务活着就返回 ok。"""
    return HealthResponse(status="ok", version=app.version)
