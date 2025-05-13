"""
WebSocket日志模块 - 提供WebSocket连接日志记录功能
"""
import logging
import uuid
from typing import Dict, Optional, Any
from fastapi import WebSocket

from .logging_config import RequestIdFilter

logger = logging.getLogger("voicerag.websocket")
perf_logger = logging.getLogger("voicerag.performance")

class WebSocketLogger:
    """WebSocket连接的日志记录器"""
    
    def __init__(self, websocket: WebSocket, user_id: Optional[str] = None):
        self.websocket = websocket
        self.connection_id = str(uuid.uuid4())
        self.user_id = user_id
        
        # 将连接ID存储在WebSocket状态中
        self.websocket.state.connection_id = self.connection_id
        
        # 创建带有连接ID的日志记录器
        self.logger = logging.LoggerAdapter(
            logger, 
            {
                "request_id": self.connection_id,
                "user_id": self.user_id or "anonymous",
                "connection_id": self.connection_id
            }
        )
        
        # 性能日志记录器
        self.perf_logger = logging.LoggerAdapter(
            perf_logger,
            {
                "request_id": self.connection_id,
                "user_id": self.user_id or "anonymous",
                "connection_id": self.connection_id
            }
        )
    
    def log_connection_open(self, client_info: Optional[Dict[str, Any]] = None):
        """记录WebSocket连接打开"""
        client_host = self.websocket.client.host if self.websocket.client else "unknown"
        self.logger.info(
            f"WebSocket connection opened: id={self.connection_id}, "
            f"user={self.user_id or 'anonymous'}, client={client_host}, "
            f"info={client_info or {}}"
        )
    
    def log_connection_close(self, code: int = 1000, reason: str = ""):
        """记录WebSocket连接关闭"""
        self.logger.info(
            f"WebSocket connection closed: id={self.connection_id}, "
            f"code={code}, reason={reason}"
        )
    
    def log_message_received(self, message_type: str, message_data: Optional[Dict[str, Any]] = None):
        """记录收到的WebSocket消息"""
        self.logger.debug(
            f"WebSocket message received: id={self.connection_id}, "
            f"type={message_type}, data={message_data or {}}"
        )
    
    def log_message_sent(self, message_type: str, message_data: Optional[Dict[str, Any]] = None):
        """记录发送的WebSocket消息"""
        self.logger.debug(
            f"WebSocket message sent: id={self.connection_id}, "
            f"type={message_type}, data={message_data or {}}"
        )
    
    def log_error(self, error_message: str, exception: Optional[Exception] = None):
        """记录WebSocket错误"""
        if exception:
            self.logger.error(
                f"WebSocket error: id={self.connection_id}, message={error_message}",
                exc_info=exception
            )
        else:
            self.logger.error(
                f"WebSocket error: id={self.connection_id}, message={error_message}"
            )
    
    def log_performance(self, operation: str, duration_ms: float, extra_data: Optional[Dict[str, Any]] = None):
        """记录WebSocket性能指标"""
        self.perf_logger.info(
            f"WebSocket performance: operation={operation}, "
            f"duration_ms={duration_ms}, "
            f"data={extra_data or {}}"
        )

# 导出
__all__ = ['WebSocketLogger']
