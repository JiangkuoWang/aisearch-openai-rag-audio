"""
配置服务模块 - 提供集中的配置管理和服务初始化功能
"""
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import openai
from fastapi import FastAPI

from .settings import Settings

logger = logging.getLogger("voicerag.config.service")

class ConfigService:
    """配置服务类，提供集中的配置管理和服务初始化功能"""

    def __init__(self, settings: Settings):
        """
        初始化配置服务

        Args:
            settings: 应用程序设置实例
        """
        self.settings = settings
        self._openai_client = None

    def setup_proxy(self) -> None:
        """设置代理环境变量"""
        self.settings.setup_proxy_environment()

    def get_openai_api_key(self) -> str:
        """
        获取OpenAI API密钥

        首先尝试从环境变量获取，然后回退到配置中的值

        Returns:
            OpenAI API密钥

        Raises:
            ValueError: 如果未设置有效的API密钥
        """
        # 尝试从环境变量获取API密钥
        openai_api_key_env = os.environ.get("OPENAI_API_KEY")
        if openai_api_key_env:
            logger.info("使用环境变量中的 OPENAI_API_KEY")
            openai_api_key = openai_api_key_env
        else:
            # 回退到配置中的API密钥
            openai_api_key = self.settings.OPENAI_API_KEY.get_secret_value()
            logger.info("使用配置中的 OPENAI_API_KEY")

        # 验证API密钥
        if not openai_api_key or openai_api_key.startswith("sk-your-"):
            error_msg = "有效的 OPENAI_API_KEY 未设置。请在 .env 文件或环境变量中设置有效的 API 密钥。"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return openai_api_key

    def get_openai_models(self) -> Tuple[str, str]:
        """
        获取OpenAI模型配置

        Returns:
            包含实时模型名称和嵌入模型名称的元组
        """
        return (
            self.settings.OPENAI_REALTIME_MODEL,
            self.settings.OPENAI_EMBEDDING_MODEL
        )

    async def init_openai_client(self) -> openai.AsyncOpenAI:
        """
        初始化OpenAI异步客户端

        Returns:
            初始化后的OpenAI异步客户端

        Raises:
            ValueError: 如果初始化失败
        """
        try:
            api_key = self.get_openai_api_key()
            http_client_kwargs = {}

            # 配置代理（如果有）
            if self.settings.ALL_PROXY:
                import httpx
                try:
                    # 创建代理客户端
                    proxy = httpx.Proxy(self.settings.ALL_PROXY)
                    transport = httpx.AsyncHTTPTransport(proxy=proxy)
                    http_client = httpx.AsyncClient(transport=transport, timeout=30.0)
                    http_client_kwargs["http_client"] = http_client
                    logger.info("OpenAI client configured with proxy")
                except Exception as e:
                    logger.warning(f"Proxy configuration failed, using direct connection: {str(e)}")

            # 初始化OpenAI客户端
            self._openai_client = openai.AsyncOpenAI(
                api_key=api_key,
                timeout=30.0,
                **http_client_kwargs
            )
            logger.info("AsyncOpenAI client initialized successfully")
            return self._openai_client
        except Exception as e:
            logger.error("Failed to initialize AsyncOpenAI client")
            raise ValueError(f"Could not initialize OpenAI client: {e}")

    async def close_openai_client(self) -> None:
        """关闭OpenAI客户端（如果已初始化）"""
        if not self._openai_client:
            logger.info("No OpenAI client to close.")
            return

        try:
            # 检查是否有异步关闭方法
            if hasattr(self._openai_client, "close") and callable(getattr(self._openai_client, "close")):
                # 检查是否是协程函数
                import asyncio
                if asyncio.iscoroutinefunction(self._openai_client.close):
                    await self._openai_client.close()
                else:
                    self._openai_client.close()
                logger.info("OpenAI client closed successfully.")
            else:
                logger.info("OpenAI client does not have a close method.")
        except Exception as e:
            logger.error(f"Error closing OpenAI client: {e}")

    def init_app(self, app: FastAPI) -> None:
        """
        初始化FastAPI应用程序状态

        Args:
            app: FastAPI应用程序实例
        """
        # 设置应用程序状态
        app.state.config_service = self

        # 设置CORS
        origins = self.settings.get_cors_origins()
        logger.info(f"CORS origins configured: {origins}")

        # 记录应用程序配置
        logger.info(f"Application initialized in {self.settings.ENVIRONMENT} environment")
        logger.info(f"Debug mode: {self.settings.DEBUG}")

    def get_rtmt_config(self) -> Dict[str, Any]:
        """
        获取RTMiddleTier配置

        Returns:
            RTMiddleTier配置字典
        """
        return {
            "openai_api_key": self.get_openai_api_key(),
            "model": self.settings.OPENAI_REALTIME_MODEL,
            "voice_choice": self.settings.OPENAI_REALTIME_VOICE_CHOICE
        }

    def get_rtmt_system_message(self) -> str:
        """
        获取RTMiddleTier系统消息

        Returns:
            系统消息字符串
        """
        return """
You are a helpful assistant. If the user has enabled the knowledge base search, only answer questions based on information found using the 'search' tool. Otherwise, answer normally.
The user is listening to answers with audio, so keep answers concise, ideally a single sentence.
Never read file names, source names, or chunk IDs out loud.
If using the knowledge base:
1. Always use the 'search' tool first.
2. Always use the 'report_grounding' tool to cite sources used.
        """.strip()

# 创建全局配置服务实例
config_service = ConfigService(Settings())
