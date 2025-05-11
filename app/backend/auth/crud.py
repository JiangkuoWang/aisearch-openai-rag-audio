import sqlite3
from typing import Optional, List
import datetime

from . import schemas # 从同级目录的 schemas.py 导入
from .security import get_password_hash # 从同级目录的 security.py 导入

# 注意：为了简化，这里的 User 模型直接使用了 schemas.User。
# 在更复杂的应用中，通常会有一个单独的 models.py 定义数据库模型 (例如 SQLAlchemy 模型)，
# CRUD 函数会返回这些模型对象，然后在API层再转换为 Pydantic schema。
# 但对于直接使用 sqlite3 和简单场景，直接使用 Pydantic schema 作为数据载体是可行的。

# 将 CRUD 函数返回类型和内部使用的模型更改为 schemas.UserInDB

def get_user(db: sqlite3.Connection, user_id: int) -> Optional[schemas.UserInDB]:
    cursor = db.cursor()
    cursor.execute(
        # 确保查询 password_hash
        "SELECT id, username, email, password_hash, is_active, created_at, last_login, role FROM users WHERE id = ?",
        (user_id,)
    )
    row = cursor.fetchone()
    if row:
        user_data = dict(zip([column[0] for column in cursor.description], row))
        return schemas.UserInDB(**user_data) # 使用 UserInDB 模型
    return None

def get_user_by_email(db: sqlite3.Connection, email: str) -> Optional[schemas.UserInDB]:
    cursor = db.cursor()
    cursor.execute(
        # 确保查询 password_hash
        "SELECT id, username, email, password_hash, is_active, created_at, last_login, role FROM users WHERE email = ?",
        (email,)
    )
    row = cursor.fetchone()
    if row:
        user_data = dict(zip([column[0] for column in cursor.description], row))
        return schemas.UserInDB(**user_data) # 使用 UserInDB 模型
    return None

def get_user_by_username(db: sqlite3.Connection, username: str) -> Optional[schemas.UserInDB]:
    cursor = db.cursor()
    cursor.execute(
        # 确保查询 password_hash
        "SELECT id, username, email, password_hash, is_active, created_at, last_login, role FROM users WHERE username = ?",
        (username,)
    )
    row = cursor.fetchone()
    if row:
        user_data = dict(zip([column[0] for column in cursor.description], row))
        return schemas.UserInDB(**user_data) # 使用 UserInDB 模型
    return None

# create_user 返回类型也更改为 UserInDB，因为其内部调用了 get_user
# FastAPI 在 API 层面会通过 response_model=schemas.User 来确保只返回 User 中的字段
def create_user(db: sqlite3.Connection, user: schemas.UserCreate) -> Optional[schemas.UserInDB]:
    hashed_password = get_password_hash(user.password)
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (user.username, user.email, hashed_password, user.role or 'user')
        )
        db.commit()
        user_id = cursor.lastrowid
        if user_id:
            # get_user 现在返回 UserInDB，所以这里也是 UserInDB
            return get_user(db, user_id) 
    except sqlite3.IntegrityError:
        db.rollback()
        return None
    return None

# 后续可以添加 update_user, delete_user, 以及与其他表相关的CRUD操作

# 用户文档相关的CRUD函数

def create_user_document(
    db: sqlite3.Connection, 
    user_id: int, 
    document: schemas.UserDocumentCreate
) -> Optional[schemas.UserDocument]:
    """
    将文档与用户关联
    
    Args:
        db: 数据库连接
        user_id: 用户ID
        document: 文档信息
        
    Returns:
        如果创建成功，返回创建的用户文档关联对象；否则返回None
    """
    cursor = db.cursor()
    try:
        # 获取当前时间字符串，用于upload_time字段
        current_time = datetime.datetime.now().isoformat()
        
        cursor.execute(
            """
            INSERT INTO user_documents 
            (user_id, document_id, is_private, custom_filename, upload_time) 
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id, 
                document.document_id, 
                1 if document.is_private else 0,  # SQLite中布尔值存储为0/1
                document.custom_filename, 
                current_time
            )
        )
        db.commit()
        
        # 获取插入的记录ID
        doc_id = cursor.lastrowid
        if doc_id:
            # 返回完整的用户文档对象
            return schemas.UserDocument(
                id=doc_id,
                user_id=user_id,
                document_id=document.document_id,
                is_private=document.is_private,
                custom_filename=document.custom_filename,
                upload_time=datetime.datetime.fromisoformat(current_time)
            )
    except sqlite3.Error as e:
        print(f"数据库错误: {e}")
        db.rollback()
    
    return None

def get_user_document(
    db: sqlite3.Connection, 
    document_id: int
) -> Optional[schemas.UserDocument]:
    """获取单个用户文档关联记录"""
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT id, user_id, document_id, is_private, custom_filename, upload_time 
        FROM user_documents WHERE id = ?
        """,
        (document_id,)
    )
    row = cursor.fetchone()
    if row:
        # 将数据库行转换为字典
        user_doc_data = dict(zip([column[0] for column in cursor.description], row))
        # 将is_private从整数转换为布尔值
        user_doc_data["is_private"] = bool(user_doc_data["is_private"])
        return schemas.UserDocument(**user_doc_data)
    return None

def get_user_documents(
    db: sqlite3.Connection, 
    user_id: int
) -> List[schemas.UserDocument]:
    """获取用户的所有文档"""
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT id, user_id, document_id, is_private, custom_filename, upload_time 
        FROM user_documents WHERE user_id = ? ORDER BY upload_time DESC
        """,
        (user_id,)
    )
    rows = cursor.fetchall()
    results = []
    
    for row in rows:
        # 将数据库行转换为字典
        user_doc_data = dict(zip([column[0] for column in cursor.description], row))
        # 将is_private从整数转换为布尔值
        user_doc_data["is_private"] = bool(user_doc_data["is_private"])
        results.append(schemas.UserDocument(**user_doc_data))
    
    return results

def delete_user_document(
    db: sqlite3.Connection, 
    document_id: int, 
    user_id: int
) -> bool:
    """
    删除用户文档关联
    
    Args:
        db: 数据库连接
        document_id: 要删除的文档ID
        user_id: 用户ID (用于确保只能删除自己的文档)
        
    Returns:
        如果删除成功，返回True；否则返回False
    """
    cursor = db.cursor()
    try:
        cursor.execute(
            "DELETE FROM user_documents WHERE id = ? AND user_id = ?",
            (document_id, user_id)
        )
        db.commit()
        # 如果影响的行数 > 0，说明删除成功
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"删除文档时出错: {e}")
        db.rollback()
        return False

def check_document_ownership(
    db: sqlite3.Connection, 
    document_id: str, 
    user_id: int
) -> bool:
    """
    检查文档是否属于特定用户
    
    Args:
        db: 数据库连接
        document_id: 文档ID
        user_id: 用户ID
        
    Returns:
        如果文档属于该用户，返回True；否则返回False
    """
    cursor = db.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM user_documents WHERE document_id = ? AND user_id = ?",
        (document_id, user_id)
    )
    count = cursor.fetchone()[0]
    return count > 0

def update_document_privacy(
    db: sqlite3.Connection, 
    document_id: int, 
    user_id: int, 
    is_private: bool
) -> bool:
    """
    更新文档的私有状态
    
    Args:
        db: 数据库连接
        document_id: 文档ID
        user_id: 用户ID (用于确保只能修改自己的文档)
        is_private: 新的私有状态
        
    Returns:
        如果更新成功，返回True；否则返回False
    """
    cursor = db.cursor()
    try:
        cursor.execute(
            "UPDATE user_documents SET is_private = ? WHERE id = ? AND user_id = ?",
            (1 if is_private else 0, document_id, user_id)
        )
        db.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"更新文档私有状态时出错: {e}")
        db.rollback()
        return False
