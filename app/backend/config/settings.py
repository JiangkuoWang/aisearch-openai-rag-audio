from typing import List, Optional, Dict, Any, Union
from pydantic import Field, field_validator, AnyHttpUrl, SecretStr
from pydantic_settings import BaseSettings
from pathlib import Path
import os
import logging

logger = logging.getLogger("voicerag")

class Settings(BaseSettings):
    # 基础配置
    APP_NAME: str = "RAG Audio Search"
    API_PREFIX: str = ""
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8765
    DEBUG: bool = False

    # 环境配置
    ENVIRONMENT: str = "development"  # development, production, testing

    # 安全配置
    SECRET_KEY: Optional[str] = Field(None, env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # CORS配置
    FRONTEND_URL: Optional[str] = None
    CORS_ORIGINS: List[str] = ["http://localhost:8765", "http://127.0.0.1:8765", "http://localhost:5173"]

    # OpenAI配置
    OPENAI_API_KEY: Optional[SecretStr] = Field(None, env="OPENAI_API_KEY")
    OPENAI_REALTIME_MODEL: str = "gpt-4o-realtime-preview"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    OPENAI_REALTIME_VOICE_CHOICE: str = "alloy"
    LLAMA_EXTRACTION_LLM_MODEL: str = "gpt-4o" # For LlamaIndex graph creation

    # 代理配置
    # 使用SOCKS5代理作为主要代理方式
    ALL_PROXY: Optional[str] = "socks5://127.0.0.1:33211"
    HTTP_PROXY: Optional[str] = None
    HTTPS_PROXY: Optional[str] = None

    # 数据库配置
    DATABASE_PATH: Path = Path("auth.db")

    # 文件存储配置
    BACKEND_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.resolve())
    STATIC_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "static")

    # RAG和脚本相关路径配置 (相对于 BACKEND_DIR)
    # 这些路径主要由 app/backend/scripts/ 下的脚本使用
    # 默认值基于之前的脚本实现
    DATA_SOURCE_DIR_RELATIVE: str = "../../data"
    RAG_DATA_DIR_RELATIVE: str = "rag_data" # General directory for RAG outputs
    LLAMA_GRAPH_INDEX_DIR_RELATIVE: str = "rag_data/llama_graph_index"
    RAG_METADATA_FILE_RELATIVE: str = "rag_data/rag_data.jsonl" # For in-memory index
    RAG_VECTOR_FILE_RELATIVE: str = "rag_data/rag_vectors.npy"   # For in-memory index


    # 验证器和辅助方法
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    @field_validator("FRONTEND_URL", mode="before")
    @classmethod
    def parse_frontend_url(cls, v: Optional[str]) -> Optional[List[str]]:
        if v is None:
            return None
        if v == "*":
            return ["*"]
        return [origin.strip() for origin in v.split(",") if origin.strip()]

    @field_validator("DEBUG", mode="before")
    @classmethod
    def set_debug_based_on_environment(cls, v: bool, info) -> bool:
        values = info.data
        if values.get("ENVIRONMENT") == "production":
            return False
        return v or True  # 在非生产环境中默认启用DEBUG

    def setup_proxy_environment(self):
        """设置代理环境变量"""
        try:
            # 优先使用ALL_PROXY
            if self.ALL_PROXY:
                os.environ["all_proxy"] = self.ALL_PROXY
                logger.info("Proxy environment variable set")
            # 仅在ALL_PROXY未设置时使用HTTP/HTTPS代理
            elif self.HTTP_PROXY or self.HTTPS_PROXY:
                if self.HTTP_PROXY:
                    os.environ["http_proxy"] = self.HTTP_PROXY
                if self.HTTPS_PROXY:
                    os.environ["https_proxy"] = self.HTTPS_PROXY
                logger.info("HTTP/HTTPS proxy environment variables set")
        except Exception as e:
            logger.warning(f"Failed to set proxy environment variables: {e}")

    def get_cors_origins(self) -> List[str]:
        """返回最终的CORS origins列表，优先使用FRONTEND_URL"""
        if self.FRONTEND_URL:
            origins = self.FRONTEND_URL
            if "*" in origins:
                return ["*"]
            return origins
        return self.CORS_ORIGINS

    model_config = {
        "env_file": str(Path(__file__).parent.parent / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore"
    }

# 日志配置
def configure_logging():
    """配置日志格式和级别"""
    log_level = logging.DEBUG if Settings().DEBUG else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)
    logger.info(f"Logging configured with level: {logging.getLevelName(log_level)}")
