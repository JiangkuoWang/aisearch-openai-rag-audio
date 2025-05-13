"""
日志配置模块 - 提供集中式日志管理功能
"""
import logging
import logging.handlers
import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import uuid
import time
from functools import wraps

# 导入配置服务，但避免循环导入
try:
    from app.backend.config import config_service
except ImportError:
    # 创建一个模拟的配置对象，仅用于日志初始化
    class MockSettings:
        DEBUG = True
        ENVIRONMENT = "development"
    
    class MockConfigService:
        settings = MockSettings()
    
    config_service = MockConfigService()

# 日志目录配置
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 日志文件路径
LOG_FILES = {
    "main": LOG_DIR / "app.log",
    "error": LOG_DIR / "error.log",
    "access": LOG_DIR / "access.log",
    "websocket": LOG_DIR / "websocket.log",
    "performance": LOG_DIR / "performance.log"
}

# 日志格式
STANDARD_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] [%(request_id)s] - %(message)s"
DETAILED_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] [%(request_id)s] [%(pathname)s:%(lineno)d] - %(message)s"
JSON_FORMAT = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "request_id": "%(request_id)s", "message": "%(message)s"}'

# 请求ID过滤器
class RequestIdFilter(logging.Filter):
    """为日志记录添加请求ID"""
    def __init__(self, request_id: Optional[str] = None):
        super().__init__()
        self.request_id = request_id or "no-request-id"
    
    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = self.request_id
        return True

# JSON格式化器
class JsonFormatter(logging.Formatter):
    """将日志记录格式化为JSON"""
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
    
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, 'request_id', 'no-request-id')
        }
        
        # 添加异常信息
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        # 添加自定义属性
        for attr, value in record.__dict__.items():
            if attr not in ["timestamp", "level", "logger", "message", "request_id", 
                           "args", "exc_info", "exc_text", "pathname", "filename", 
                           "module", "lineno", "funcName", "created", "msecs", 
                           "relativeCreated", "levelname", "levelno", "msg"]:
                try:
                    # 尝试将值转换为JSON可序列化的格式
                    json.dumps({attr: value})
                    log_record[attr] = value
                except (TypeError, OverflowError):
                    log_record[attr] = str(value)
        
        return json.dumps(log_record)

# 性能日志记录器
class PerformanceLogger:
    """用于记录性能指标的工具类"""
    def __init__(self, logger_name="voicerag.performance"):
        self.logger = logging.getLogger(logger_name)
        self.timers = {}
    
    def start_timer(self, operation_name: str, request_id: Optional[str] = None) -> str:
        """开始计时操作"""
        timer_id = f"{operation_name}_{uuid.uuid4().hex[:8]}"
        self.timers[timer_id] = {
            "start": time.time(),
            "operation": operation_name,
            "request_id": request_id
        }
        return timer_id
    
    def stop_timer(self, timer_id: str, extra_data: Optional[Dict[str, Any]] = None) -> float:
        """停止计时并记录性能日志"""
        if timer_id not in self.timers:
            self.logger.warning(f"Timer {timer_id} not found")
            return 0
        
        timer_data = self.timers.pop(timer_id)
        elapsed = time.time() - timer_data["start"]
        
        log_data = {
            "operation": timer_data["operation"],
            "duration_ms": round(elapsed * 1000, 2),
            **(extra_data or {})
        }
        
        # 创建一个临时的日志记录器，添加请求ID
        tmp_logger = logging.LoggerAdapter(self.logger, {"request_id": timer_data.get("request_id", "no-request-id")})
        tmp_logger.info(f"Performance: {json.dumps(log_data)}")
        
        return elapsed
    
    def log_metric(self, metric_name: str, value: Any, request_id: Optional[str] = None, **kwargs):
        """记录任意性能指标"""
        log_data = {
            "metric": metric_name,
            "value": value,
            **kwargs
        }
        
        # 创建一个临时的日志记录器，添加请求ID
        tmp_logger = logging.LoggerAdapter(self.logger, {"request_id": request_id or "no-request-id"})
        tmp_logger.info(f"Metric: {json.dumps(log_data)}")

# 性能计时装饰器
def log_performance(operation_name: Optional[str] = None):
    """记录函数执行时间的装饰器"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 获取操作名称
            op_name = operation_name or func.__name__
            
            # 获取请求ID（如果在kwargs中）
            request_id = kwargs.get('request_id', 'no-request-id')
            
            # 开始计时
            perf_logger = PerformanceLogger()
            timer_id = perf_logger.start_timer(op_name, request_id)
            
            try:
                # 执行原函数
                result = await func(*args, **kwargs)
                return result
            finally:
                # 停止计时并记录
                perf_logger.stop_timer(timer_id)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 获取操作名称
            op_name = operation_name or func.__name__
            
            # 获取请求ID（如果在kwargs中）
            request_id = kwargs.get('request_id', 'no-request-id')
            
            # 开始计时
            perf_logger = PerformanceLogger()
            timer_id = perf_logger.start_timer(op_name, request_id)
            
            try:
                # 执行原函数
                result = func(*args, **kwargs)
                return result
            finally:
                # 停止计时并记录
                perf_logger.stop_timer(timer_id)
        
        # 根据函数类型选择适当的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

# 配置日志系统
def configure_logging(env: str = None, debug: bool = None):
    """配置日志系统"""
    # 使用传入的参数或从配置服务获取
    environment = env or config_service.settings.ENVIRONMENT
    is_debug = debug if debug is not None else config_service.settings.DEBUG
    
    # 设置日志级别
    log_level = logging.DEBUG if is_debug else logging.INFO
    
    # 创建根记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加请求ID过滤器
    request_id_filter = RequestIdFilter()
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = DETAILED_FORMAT if is_debug else STANDARD_FORMAT
    console_handler.setFormatter(logging.Formatter(console_format))
    console_handler.addFilter(request_id_filter)
    root_logger.addHandler(console_handler)
    
    # 主日志文件处理器（轮转）
    main_file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILES["main"], maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    main_file_handler.setLevel(log_level)
    main_file_handler.setFormatter(logging.Formatter(STANDARD_FORMAT))
    main_file_handler.addFilter(request_id_filter)
    root_logger.addHandler(main_file_handler)
    
    # 错误日志文件处理器（轮转）
    error_file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILES["error"], maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(logging.Formatter(DETAILED_FORMAT))
    error_file_handler.addFilter(request_id_filter)
    root_logger.addHandler(error_file_handler)
    
    # 访问日志处理器
    access_logger = logging.getLogger("voicerag.access")
    access_handler = logging.handlers.TimedRotatingFileHandler(
        LOG_FILES["access"], when='midnight', backupCount=30, encoding='utf-8'
    )
    access_handler.setLevel(logging.INFO)
    access_handler.setFormatter(JsonFormatter())
    access_handler.addFilter(request_id_filter)
    access_logger.addHandler(access_handler)
    
    # WebSocket日志处理器
    ws_logger = logging.getLogger("voicerag.websocket")
    ws_handler = logging.handlers.RotatingFileHandler(
        LOG_FILES["websocket"], maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    ws_handler.setLevel(log_level)
    ws_handler.setFormatter(JsonFormatter())
    ws_handler.addFilter(request_id_filter)
    ws_logger.addHandler(ws_handler)
    
    # 性能日志处理器
    perf_logger = logging.getLogger("voicerag.performance")
    perf_handler = logging.handlers.RotatingFileHandler(
        LOG_FILES["performance"], maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(JsonFormatter())
    perf_handler.addFilter(request_id_filter)
    perf_logger.addHandler(perf_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    
    # 生产环境特殊配置
    if environment == "production":
        # 禁用调试日志
        console_handler.setLevel(logging.INFO)
        # 添加邮件处理器用于严重错误
        if os.environ.get("SMTP_HOST"):
            mail_handler = logging.handlers.SMTPHandler(
                mailhost=os.environ.get("SMTP_HOST"),
                fromaddr=os.environ.get("SMTP_FROM", "noreply@example.com"),
                toaddrs=os.environ.get("SMTP_TO", "admin@example.com").split(","),
                subject=f"[{config_service.settings.APP_NAME}] Critical Error"
            )
            mail_handler.setLevel(logging.CRITICAL)
            mail_handler.setFormatter(logging.Formatter(DETAILED_FORMAT))
            root_logger.addHandler(mail_handler)
    
    # 记录初始化完成
    logging.getLogger("voicerag").info(
        f"Logging system initialized: environment={environment}, debug={is_debug}, "
        f"level={logging.getLevelName(log_level)}"
    )

# 导出工具函数和类
__all__ = [
    'configure_logging',
    'RequestIdFilter',
    'JsonFormatter',
    'PerformanceLogger',
    'log_performance',
    'LOG_DIR',
    'LOG_FILES'
]
