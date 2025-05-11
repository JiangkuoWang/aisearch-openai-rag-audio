from pydantic import BaseModel, EmailStr
from typing import Optional, List
import datetime

# 基础用户模型，包含所有用户通用的字段
class UserBase(BaseModel):
    username: str
    email: EmailStr
    role: Optional[str] = 'user'


# 创建用户时使用的模型，继承自UserBase，并添加password字段
class UserCreate(UserBase):
    password: str


# 从数据库读取或API返回用户数据时使用的模型
class User(UserBase):
    id: int
    is_active: bool = True
    created_at: datetime.datetime
    last_login: Optional[datetime.datetime] = None

    class Config:
        # Pydantic V2 使用 from_attributes = True 替代 orm_mode = True
        # 用于告知 Pydantic 模型可以从 ORM 对象（或任何具有属性的对象）中读取数据
        from_attributes = True


# 新增：用于数据库内部操作的模型，包含 password_hash
class UserInDB(User): # User 包含了 id, is_active, created_at, last_login, role, email, username
    password_hash: str


# 用于令牌的数据模型 (后续步骤会用到)
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


# 用户文档关联的模型
class UserDocumentBase(BaseModel):
    document_id: str  # 文档ID
    is_private: bool = True  # 默认为私有文档
    custom_filename: Optional[str] = None  # 用户自定义的文件名

class UserDocumentCreate(UserDocumentBase):
    pass  # 创建时使用基础模型的所有字段

class UserDocument(UserDocumentBase):
    id: int  # 数据库中的主键ID
    user_id: int  # 关联的用户ID
    upload_time: datetime.datetime  # 上传时间
    
    class Config:
        from_attributes = True

# 密码更新模型
class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str
