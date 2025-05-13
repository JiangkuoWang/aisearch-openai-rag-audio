import logging
import os
from pathlib import Path
from .settings import Settings

# 配置日志
logger = logging.getLogger("voicerag")

try:
    # 检查 .env 文件是否存在
    env_file_path = Path(__file__).parent.parent / ".env"
    if env_file_path.exists():
        logger.info(f".env 文件存在于: {env_file_path}")
        # 打印 .env 文件的前几行（不包含敏感信息）
        with open(env_file_path, "r") as f:
            env_content = f.readlines()
            logger.info(f".env 文件包含 {len(env_content)} 行")
            # 打印不包含敏感信息的行
            for line in env_content:
                if not any(key in line for key in ["SECRET_KEY", "API_KEY", "PASSWORD"]):
                    logger.info(f"ENV: {line.strip()}")
    else:
        logger.warning(f".env 文件不存在于: {env_file_path}")

    # 检查环境变量
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        logger.info("从环境变量中找到 OPENAI_API_KEY")
    else:
        logger.warning("环境变量中没有找到 OPENAI_API_KEY")

    # 尝试手动加载 .env 文件
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file_path)
        logger.info(f"手动加载 .env 文件: {env_file_path}")
        # 再次检查环境变量
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            logger.info("加载 .env 文件后找到 OPENAI_API_KEY")
    except ImportError:
        logger.warning("无法导入 dotenv 模块，跳过手动加载 .env 文件")
    except Exception as e:
        logger.warning(f"手动加载 .env 文件失败: {e}")

    # 创建全局设置实例
    settings = Settings()

    # 打印部分配置（不包含敏感信息）
    logger.info(f"环境: {settings.ENVIRONMENT}")
    logger.info(f"调试模式: {settings.DEBUG}")
    logger.info(f"后端主机: {settings.BACKEND_HOST}")
    logger.info(f"后端端口: {settings.BACKEND_PORT}")

    # 设置代理环境变量
    settings.setup_proxy_environment()

    logger.info("配置模块初始化成功")
except Exception as e:
    logger.error(f"配置模块初始化失败: {e}")
    # 创建一个基本的设置实例，避免导入错误
    settings = Settings(
        SECRET_KEY="fallback-secret-key-for-emergency-use-only",
        OPENAI_API_KEY="sk-fallback-key-for-emergency"
    )
    logger.warning("使用了应急配置，请检查环境变量和.env文件")

# 导出设置实例，方便其他模块导入
__all__ = ["settings"]
