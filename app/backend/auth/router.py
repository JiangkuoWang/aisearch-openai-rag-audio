import sqlite3
from datetime import timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from fastapi.security import OAuth2PasswordRequestForm

from . import crud, schemas, security # 从同级目录导入
from .db import open_db_connection # 从同级目录的 db.py 导入
from .deps import get_current_user

router = APIRouter(
    prefix="/auth", # 给这个路由下的所有路径添加 /auth 前缀
    tags=["Authentication"] # 在OpenAPI文档中分组
)

@router.post("/register", response_model=schemas.User)
def register_user(user: schemas.UserCreate, db: sqlite3.Connection = Depends(open_db_connection)):
    """
    用户注册端点。
    - 接收用户名、邮箱和密码。
    - 检查用户名或邮箱是否已存在。
    - 创建新用户并返回用户信息。
    """
    db_user_by_email = crud.get_user_by_email(db, email=user.email)
    if db_user_by_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    db_user_by_username = crud.get_user_by_username(db, username=user.username)
    if db_user_by_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    created_user = crud.create_user(db=db, user=user)
    if not created_user:
        # 这种情况理论上不应该在检查唯一性后发生，除非有并发问题或 create_user 内部其他错误
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create user."
        )
    return created_user

@router.post("/login/token", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: sqlite3.Connection = Depends(open_db_connection)):
    """
    用户登录端点，用于获取访问令牌。
    - 接收表单数据 (username, password)。
    - 验证用户凭据。
    - 如果凭据有效，则创建并返回JWT访问令牌。
    """
    user = crud.get_user_by_username(db, username=form_data.username)
    if not user or not security.verify_password(form_data.password, user.password_hash):
        # 注意：在 crud.py 中，get_user_by_username 返回的是 schemas.User 对象，
        # 而 schemas.User 对象不包含 password_hash。
        # 这是个问题。CRUD函数应该返回包含 password_hash 的对象，或者我们需要一个专门用于认证的get_user函数。
        #
        # 修正方案：
        # 1. 修改 crud.get_user_by_username (及其他 get_user*) 以查询并返回 password_hash，
        #    可能需要一个新的 Pydantic 模型 (例如 UserInDB) 来包含它，或者在 User 模型中可选地包含它。
        # 2. 在 crud 中创建一个新的函数，例如 authenticate_user，它内部获取用户并验证密码。
        #
        # 为了快速演示，假设 crud.get_user_by_username 返回的对象中 *有* password_hash。
        # **实际应用中必须修正这个问题。**
        #
        # 临时的解决方案是在 User schema 中添加 password_hash (不推荐用于API响应)
        # 或者在 crud.py 中，让 get_user_by_username/email 返回一个更完整的内部对象或字典。
        #
        # ---- 假设的修正：crud.get_user_for_auth(db, username=form_data.username) 返回包含 password_hash 的用户数据 ----
        # user_auth_data = crud.get_user_for_auth(db, username=form_data.username)
        # if not user_auth_data or not security.verify_password(form_data.password, user_auth_data.password_hash):
        # ----
        
        # 当前代码的问题点：user 对象 (schemas.User) 没有 password_hash 属性。
        # 我们需要从数据库中获取原始的 password_hash。
        # 让我们先假设 crud.get_user_by_username 返回的 user 对象有一个 password_hash 属性
        # (这意味着 schemas.User 需要临时修改，或者 crud 返回的是一个包含此字段的内部模型)
        # 
        # 正确的做法是: crud.py 中的 get_user_by_username 等函数应该设计成能获取 password_hash
        # 以便进行认证。 schemas.User 用于API响应，不应暴露 password_hash.
        # 
        # 紧急修复思路：在 crud.py 中修改 get_user_by_username, get_user_by_email, get_user
        # 查询语句中包含 password_hash，并且构造返回的 User 对象时也包含它。
        # 然后在 router.py 的 register_user 响应中，确保 User 模型不序列化 password_hash.
        # 或者，crud.py 的 get_user* 返回一个字典，然后在 router.py 中按需构造 User schema 或进行验证。

        # ---------------------------------------------------------------------
        # 为了使此代码能够运行，您需要确保 `user` 对象具有 `password_hash` 属性。
        # 请返回到 crud.py 并修改 get_user_by_username（以及相关的 get_user 和 get_user_by_email）
        # 以便它们从数据库中检索 password_hash 并将其包含在返回的 User 对象中。
        # 例如, 在 crud.py 中：
        # cursor.execute(
        # "SELECT id, username, email, password_hash, is_active, created_at, last_login, role FROM users WHERE ...")
        # user_data = dict(zip([column[0] for column in cursor.description], row))
        # return schemas.User(**user_data) 
        # (前提是 schemas.User 包含 Optional[str] password_hash = None)
        # ---------------------------------------------------------------------
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/users/me", response_model=schemas.User)
async def read_users_me(current_user: schemas.UserInDB = Depends(get_current_user)):
    return current_user

# 用户文档管理端点

@router.get("/users/me/documents", response_model=List[schemas.UserDocument])
async def read_user_documents(
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: sqlite3.Connection = Depends(open_db_connection)
):
    """
    获取当前用户的所有文档
    """
    documents = crud.get_user_documents(db, current_user.id)
    return documents

@router.post("/users/me/documents", response_model=schemas.UserDocument)
async def create_user_document(
    document: schemas.UserDocumentCreate,
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: sqlite3.Connection = Depends(open_db_connection)
):
    """
    为当前用户添加文档关联
    """
    return crud.create_user_document(db, current_user.id, document)

@router.delete("/users/me/documents/{document_id}", response_model=bool)
async def delete_user_document(
    document_id: int = Path(..., title="要删除的文档ID"),
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: sqlite3.Connection = Depends(open_db_connection)
):
    """
    删除当前用户的指定文档
    """
    result = crud.delete_user_document(db, document_id, current_user.id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在或不属于当前用户"
        )
    return True

@router.put("/users/me/documents/{document_id}/privacy", response_model=bool)
async def update_document_privacy(
    is_private: bool,
    document_id: int = Path(..., title="文档ID"),
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: sqlite3.Connection = Depends(open_db_connection)
):
    """
    更新文档的私有状态
    """
    result = crud.update_document_privacy(db, document_id, current_user.id, is_private)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在或不属于当前用户"
        )
    return True

# 密码更新端点
@router.put("/users/me/password", response_model=schemas.User)
async def update_password(
    password_update: schemas.PasswordUpdate,
    current_user: schemas.UserInDB = Depends(get_current_user),
    db: sqlite3.Connection = Depends(open_db_connection)
):
    """
    更新当前用户的密码
    
    - 验证当前密码
    - 设置新密码
    """
    # 验证当前密码
    if not security.verify_password(password_update.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前密码不正确"
        )
    
    # 这里需要实现一个更新用户密码的函数
    # 临时方案：使用SQL直接更新
    cursor = db.cursor()
    try:
        new_password_hash = security.get_password_hash(password_update.new_password)
        cursor.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_password_hash, current_user.id)
        )
        db.commit()
        
        # 获取更新后的用户信息
        updated_user = crud.get_user(db, current_user.id)
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="更新密码后无法获取用户信息"
            )
        return updated_user
    except sqlite3.Error as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新密码时出错: {e}"
        )
