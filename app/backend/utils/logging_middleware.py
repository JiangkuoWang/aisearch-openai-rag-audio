"""
日志中间件模块 - 提供HTTP请求日志记录功能
"""
import time
import uuid
import logging
from typing import Callable, Dict, Any
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .logging_config import RequestIdFilter

logger = logging.getLogger("voicerag.access")
perf_logger = logging.getLogger("voicerag.performance")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    中间件，用于记录请求和响应信息，并添加请求ID
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        # 记录请求开始
        start_time = time.time()
        
        # 提取请求信息
        path = request.url.path
        method = request.method
        client_host = request.client.host if request.client else "unknown"
        
        # 创建请求日志记录器（带请求ID）
        request_logger = logging.LoggerAdapter(logger, {"request_id": request_id})
        
        # 记录请求信息
        request_logger.info(
            f"Request started: {method} {path} from {client_host}"
        )
        
        # 将请求ID添加到请求状态中
        request.state.request_id = request_id
        
        # 处理请求
        try:
            # 添加请求ID过滤器到根记录器
            root_logger = logging.getLogger()
            request_id_filter = RequestIdFilter(request_id)
            
            # 保存原始过滤器
            original_filters = root_logger.filters.copy()
            
            # 清除现有过滤器并添加新过滤器
            for f in root_logger.filters:
                root_logger.removeFilter(f)
            root_logger.addFilter(request_id_filter)
            
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录响应信息
            status_code = response.status_code
            request_logger.info(
                f"Request completed: {method} {path} - {status_code} in {process_time:.3f}s"
            )
            
            # 记录性能指标
            perf_log = logging.LoggerAdapter(perf_logger, {"request_id": request_id})
            perf_log.info(
                f"API request: path={path}, method={method}, status={status_code}, "
                f"duration_ms={round(process_time * 1000, 2)}"
            )
            
            # 在响应头中添加请求ID（可选）
            response.headers["X-Request-ID"] = request_id
            
            return response
        except Exception as e:
            # 记录异常
            process_time = time.time() - start_time
            request_logger.error(
                f"Request failed: {method} {path} - Exception: {str(e)} in {process_time:.3f}s",
                exc_info=True
            )
            raise
        finally:
            # 恢复原始过滤器
            for f in root_logger.filters:
                root_logger.removeFilter(f)
            for f in original_filters:
                root_logger.addFilter(f)

def setup_request_logging(app: FastAPI) -> None:
    """设置请求日志记录中间件"""
    app.add_middleware(RequestLoggingMiddleware)
