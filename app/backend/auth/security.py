from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone # 确保导入 timezone
from typing import Optional

from jose import JWTError, jwt

from . import schemas # 从同级目录导入 schemas，用于 TokenData
try:
    # 当作为包导入时
    from app.backend.config import config_service # 导入配置服务
except ModuleNotFoundError:
    # 当直接运行时
    import sys
    import os
    # 添加项目根目录到Python路径
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from config import config_service # 导入配置服务


# 配置密码哈希上下文，推荐使用 bcrypt
# schemes 定义了支持的哈希算法列表，deprecated="auto" 会自动处理旧算法的迁移（如果需要）
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证明文密码与哈希密码是否匹配。

    :param plain_password: 用户输入的明文密码。
    :param hashed_password: 数据库中存储的哈希密码。
    :return: 如果密码匹配则返回 True，否则返回 False。
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    生成密码的哈希值。

    :param password: 用户输入的明文密码。
    :return: 生成的密码哈希字符串。
    """
    return pwd_context.hash(password)

# 可以在这里添加其他与安全相关的工具函数，例如创建和验证JWT令牌等（后续步骤）


# --- JWT 令牌部分 --- #

# 从配置服务加载安全配置
SECRET_KEY = config_service.settings.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = config_service.settings.ACCESS_TOKEN_EXPIRE_MINUTES

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    创建JWT访问令牌。

    :param data: 要编码到令牌中的数据 (例如: {"sub": username})。
    :param expires_delta: 可选的过期时间增量。如果未提供，则使用默认值。
    :return:编码后的JWT字符串。
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 这个函数用于验证令牌并提取数据，我们将在后续实现受保护路由时使用它。
# 现在先定义一个基本框架。
def verify_token_and_get_data(token: str, credentials_exception) -> Optional[schemas.TokenData]: # credentials_exception 将是一个 FastAPI HTTPExceptio
    """
    验证JWT令牌，如果有效则返回令牌中的数据 (payload)。
    如果令牌无效或过期，则引发提供的 credentials_exception。
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception # subject (用户名) 不存在
        # 可以添加更多验证，例如 token_id, scope 等
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception # JWT解码失败或任何JWT相关的错误
    return token_data

# 可以在这里添加刷新令牌 (refresh token) 的逻辑，如果需要的话。