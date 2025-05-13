"""
异常处理工具模块 - 提供统一的异常处理机制
"""
import functools
import logging
import traceback
from typing import Any, Callable, TypeVar, cast, Optional, Dict, Union

import openai
from fastapi import HTTPException, WebSocket, status
from jose import JWTError, ExpiredSignatureError

# 类型变量，用于装饰器返回类型
T = TypeVar('T', bound=Callable[..., Any])

# 获取模块日志记录器
logger = logging.getLogger("voicerag.error_handlers")

class AppError(Exception):
    """应用程序自定义异常基类"""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


def api_exception_handler(func: T) -> T:
    """
    API端点异常处理装饰器，用于FastAPI路由函数
    
    将捕获所有异常并转换为适当的HTTPException，同时记录详细日志
    
    Args:
        func: 要装饰的API函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # 直接重新抛出HTTP异常，保留原始状态码和详情
            raise
        except AppError as e:
            # 处理应用程序自定义异常
            logger.error(f"Application error in {func.__name__}: {e.message}")
            if e.details:
                logger.error(f"Error details: {e.details}")
            raise HTTPException(status_code=e.status_code, detail=e.message)
        except openai.RateLimitError as e:
            # 处理OpenAI速率限制异常
            logger.warning(f"OpenAI rate limit exceeded in {func.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="API rate limit exceeded. Please try again later."
            )
        except openai.APIError as e:
            # 处理OpenAI API异常
            logger.error(f"OpenAI API error in {func.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Error communicating with OpenAI API: {str(e)}"
            )
        except JWTError as e:
            # 处理JWT相关异常
            logger.warning(f"JWT validation error in {func.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except Exception as e:
            # 处理所有其他未预期的异常
            logger.exception(f"Unhandled exception in {func.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error: {str(e)}"
            )
    
    return cast(T, wrapper)


def websocket_exception_handler(func: T) -> T:
    """
    WebSocket端点异常处理装饰器
    
    捕获WebSocket处理过程中的异常，记录日志并确保WebSocket正确关闭
    
    Args:
        func: 要装饰的WebSocket处理函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    async def wrapper(websocket: WebSocket, *args: Any, **kwargs: Any) -> Any:
        try:
            return await func(websocket, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in WebSocket handler {func.__name__}: {e}")
            try:
                # 检查WebSocket是否已连接，如果是则关闭
                if websocket.client_state == WebSocket.CLIENT_STATE_CONNECTED:
                    await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            except Exception as close_error:
                logger.error(f"Error closing WebSocket: {close_error}")
    
    return cast(T, wrapper)


def openai_retry_handler(max_retries: int = 3, initial_delay: float = 1.0):
    """
    OpenAI API调用重试装饰器
    
    为OpenAI API调用添加指数退避重试逻辑
    
    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟时间（秒）
        
    Returns:
        装饰器函数
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            import asyncio
            
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except openai.RateLimitError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Rate limit exceeded, retrying in {delay} seconds... "
                            f"(Attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        delay *= 2  # 指数退避
                except (openai.APIError, openai.APIConnectionError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"OpenAI API error, retrying in {delay} seconds... "
                            f"(Attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        await asyncio.sleep(delay)
                        delay *= 2  # 指数退避
            
            # 如果所有重试都失败，抛出最后一个异常
            if last_exception:
                logger.error(f"All {max_retries} retry attempts failed: {last_exception}")
                raise last_exception
            
            # 这行代码理论上不会执行，因为如果所有重试都失败，上面的代码会抛出异常
            return None
        
        return cast(T, wrapper)
    
    return decorator


def log_function_call(level: Union[int, str] = logging.DEBUG):
    """
    函数调用日志装饰器
    
    记录函数的调用参数和返回值
    
    Args:
        level: 日志级别
        
    Returns:
        装饰器函数
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__
            logger.log(level, f"Calling {func_name} with args: {args}, kwargs: {kwargs}")
            try:
                result = await func(*args, **kwargs)
                logger.log(level, f"{func_name} returned: {result}")
                return result
            except Exception as e:
                logger.log(level, f"{func_name} raised exception: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__
            logger.log(level, f"Calling {func_name} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func_name} returned: {result}")
                return result
            except Exception as e:
                logger.log(level, f"{func_name} raised exception: {e}")
                raise
        
        # 根据原函数是否为异步函数选择适当的包装器
        if asyncio.iscoroutinefunction(func):
            return cast(T, async_wrapper)
        return cast(T, sync_wrapper)
    
    return decorator
