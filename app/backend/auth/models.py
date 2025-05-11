"""
认证系统模型和帮助函数
"""
import sqlite3
from typing import Optional, Dict, Any, List
import jwt
# import logging
from datetime import datetime
from . import crud, schemas
from .security import SECRET_KEY, ALGORITHM

def authenticate_token(token: str, db_connection: sqlite3.Connection) -> Optional[schemas.UserInDB]:
    """
    验证JWT令牌并返回用户对象
    
    参数:
        token: JWT令牌字符串（不含"Bearer "前缀）
        db_connection: SQLite数据库连接
        
    返回:
        如果令牌有效，返回用户对象；否则返回None
    """
    try:
        # 解码JWT令牌
        print(f"Authenticating token. SECRET_KEY in models: {SECRET_KEY}") # 添加日志
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return None
            
        # 从数据库获取用户
        user = crud.get_user_by_username(db_connection, username)
        if user:
            print(f"User '{username}' found in DB by authenticate_token: ID {user.id}")
        else:
            print(f"User '{username}' NOT FOUND in DB by authenticate_token!") # 关键日志
        
        return user
    except jwt.PyJWTError:
        return None
    except Exception as e:
        print(f"验证令牌时出错: {e}")
        return None

def associate_document_with_user(
    db_connection: sqlite3.Connection,
    user_id: int,
    document_id: str,
    custom_filename: Optional[str] = None,
    is_private: bool = True
) -> bool:
    """
    将文档与用户关联的工具函数
    
    参数:
        db_connection: SQLite数据库连接
        user_id: 用户ID
        document_id: 文档ID
        custom_filename: 自定义文件名（可选）
        is_private: 是否为私有文档
        
    返回:
        如果关联成功，返回True；否则返回False
    """
    document = schemas.UserDocumentCreate(
        document_id=document_id,
        custom_filename=custom_filename,
        is_private=is_private
    )
    
    result = crud.create_user_document(db_connection, user_id, document)
    return result is not None

def get_user_document_ids(
    db_connection: sqlite3.Connection,
    user_id: int
) -> List[str]:
    """
    获取用户的所有文档ID
    
    参数:
        db_connection: SQLite数据库连接
        user_id: 用户ID
        
    返回:
        文档ID列表
    """
    documents = crud.get_user_documents(db_connection, user_id)
    return [doc.document_id for doc in documents]
