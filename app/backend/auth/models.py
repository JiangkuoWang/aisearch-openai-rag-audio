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
