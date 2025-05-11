import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.backend.auth.router import router as auth_router
import os
from pathlib import Path
import sys

# 添加：直接导入并运行setup_database的main函数
# 这会确保在启动测试服务器之前数据库已经被创建
print("正在初始化数据库...")
from app.backend.scripts.setup_database import main as setup_db

# 执行数据库初始化
setup_db()

# 添加：打印当前数据库文件的位置以便确认
from app.backend.auth.db import DB_FILE
print(f"使用的数据库文件路径: {DB_FILE}")

app = FastAPI(title="认证系统测试")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:8765", "http://localhost:3000"],  # 允许的前端域名
    allow_credentials=True,  # 允许携带凭证（cookies等）
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有headers
)

# 注册认证路由
app.include_router(auth_router)

# 添加一个简单的根路由，方便检查API是否正常运行
@app.get("/")
async def root():
    return {
        "message": "认证系统测试服务器正在运行",
        "endpoints": {
            "用户注册": "/auth/register",
            "用户登录": "/auth/login/token",
            "当前用户": "/auth/users/me",
            "用户文档列表": "/auth/users/me/documents",
            "添加文档": "/auth/users/me/documents",
            "删除文档": "/auth/users/me/documents/{document_id}",
            "更新文档私有状态": "/auth/users/me/documents/{document_id}/privacy",
            "修改密码": "/auth/users/me/password"
        },
        "swagger文档": "/docs"
    }

if __name__ == "__main__":
    print("启动认证系统测试服务器...")
    print("测试端点:")
    print("  - 注册: http://127.0.0.1:8766/auth/register")
    print("  - 登录: http://127.0.0.1:8766/auth/login/token")
    print("  - 当前用户: http://127.0.0.1:8766/auth/users/me")
    print("  - 用户文档: http://127.0.0.1:8766/auth/users/me/documents")
    print("  - API文档: http://127.0.0.1:8766/docs")
    uvicorn.run(app, host="127.0.0.1", port=8766) 