"""
配置模块 - 提供应用程序配置和服务
"""
import logging
import os
from pathlib import Path

# 配置日志
logger = logging.getLogger("voicerag")

# 尝试加载.env文件
env_file_path = Path(__file__).parent.parent / ".env"
try:
    from dotenv import load_dotenv
    if env_file_path.exists():
        load_dotenv(dotenv_path=env_file_path)
        logger.info(f"已加载.env文件: {env_file_path}")
    else:
        logger.warning(f".env文件不存在: {env_file_path}")
except ImportError:
    logger.warning("无法导入dotenv模块，跳过.env文件加载")
except Exception as e:
    logger.warning(f"加载.env文件失败: {e}")

# 导入配置服务
try:
    from .service import config_service

    # 设置代理环境变量
    config_service.setup_proxy()

    logger.info("配置模块初始化成功")
except Exception as e:
    logger.error(f"配置模块初始化失败: {e}")
    # 导入设置类以创建应急配置
    from .settings import Settings
    from .service import ConfigService

    # 创建应急配置
    emergency_settings = Settings(
        SECRET_KEY="fallback-secret-key-for-emergency-use-only",
        OPENAI_API_KEY="sk-fallback-key-for-emergency"
    )
    config_service = ConfigService(emergency_settings)
    logger.warning("使用了应急配置，请检查环境变量和.env文件")

# 导出配置服务实例，方便其他模块导入
__all__ = ["config_service"]
