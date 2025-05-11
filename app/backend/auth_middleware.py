"""
aiohttp认证中间件
用于在aiohttp应用中验证JWT令牌
"""
import sqlite3
from aiohttp import web
from aiohttp.web import middleware
import re
import logging
import json
from typing import Optional, Callable, Dict, Any, Tuple

# Local imports
from auth.models import authenticate_token
# 修改导入，直接使用新的 open_db_connection
from auth.db import open_db_connection # <--- 修改这里

logger = logging.getLogger(__name__)

# 定义不需要认证的路径
PUBLIC_PATHS = [
    r'^/auth/.*$',  # 认证相关的端点
    r'^/realtime$',  # WebSocket端点
    r'^/static/.*$',  # 静态文件
    r'^/assets/.*$',  # 前端静态资源文件
    r'^/$',          # 根路径 (如果有首页)
    r'^/rag-config$',  # RAG配置端点
    # r'^/auth-status$', # 考虑是否公开 /auth-status
]

def is_public_path(path: str) -> bool:
    """检查路径是否为公开访问路径"""
    return any(re.match(pattern, path) for pattern in PUBLIC_PATHS)

def get_token_from_request(request) -> Optional[str]:
    """从请求头或查询参数中提取令牌"""
    # 从Authorization头提取
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header[7:]  # 移除 "Bearer " 前缀
    
    # 从查询参数提取 (用于WebSocket)
    token = request.query.get('access_token')
    if token:
        return token
    
    return None

# get_db 函数不再需要，因为中间件会直接管理连接
# def get_db() -> sqlite3.Connection:
#     """获取数据库连接（非生成器版本）"""
#     # 注意：这里我们使用的是简单的函数，而不是FastAPI的依赖注入生成器
#     # 我们需要手动管理连接的关闭
#     db_gen = open_db_connection()
#     return next(db_gen)  # 从生成器获取连接

@middleware
async def jwt_auth_middleware(request: web.Request, handler) -> web.Response:
    """
    JWT认证中间件
    验证请求中的令牌，如果有效则将用户信息添加到请求对象中
    """
    # 检查是否为公开路径
    if is_public_path(request.path):
        return await handler(request)
    
    # 提取令牌
    token = get_token_from_request(request)
    if not token:
        return web.json_response(
            {"error": "未提供认证令牌"}, 
            status=401,
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    db_conn: Optional[sqlite3.Connection] = None # 明确类型
    try:
        db_conn = open_db_connection() # <--- 直接打开连接
        user = authenticate_token(token, db_conn) # 使用打开的连接
        
        if not user:
            # 注意：如果 authenticate_token 内部因为 SECRET_KEY 或其他 jwt 问题返回 None
            # db_conn 仍然会在 finally 中被关闭
            return web.json_response(
                {"error": "无效或过期的令牌"}, 
                status=401,
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # 将用户信息添加到请求对象中
        request["user"] = user
        request["user_id"] = user.id
        
        # 调用下一个处理程序
        return await handler(request)
    except sqlite3.Error as db_err: # 更具体地捕获数据库操作错误
        logger.exception(f"数据库操作认证过程中出错: {db_err}")
        return web.json_response({"error": "服务器数据库错误"}, status=500)
    except Exception as e:
        # 此处的异常可能是 jwt.decode 失败（如果 authenticate_token 内部不捕获所有异常）
        # 或者其他意外错误
        logger.exception(f"认证过程中出错: {e}")
        return web.json_response(
            {"error": "服务器认证错误"}, 
            status=500
        )
    finally:
        # 确保关闭数据库连接
        if db_conn:
            logger.debug(f"Closing DB connection in jwt_auth_middleware for request: {request.path}")
            db_conn.close() # <--- 中间件在其 finally 块中关闭连接

def setup_auth_middleware(app: web.Application) -> None:
    """
    在aiohttp应用中设置认证中间件
    """
    app.middlewares.append(jwt_auth_middleware)
    logger.info("已添加JWT认证中间件 (修改版)") 