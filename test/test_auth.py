import sys
import os

# Add the project root to sys.path to allow importing 'app'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
"""
本测试脚本 (`test_auth.py`) 使用 PyTest 框架，旨在全面测试 FastAPI 后端应用的认证模块。

主要测试目标包括：
1.  用户注册功能：
    - 成功注册新用户。
    - 处理用户名或邮箱已存在等重复注册场景。
2.  用户登录功能（基于 Token）：
    - 使用正确凭据成功登录并获取访问令牌。
    - 处理密码错误或用户不存在等登录失败场景。
3.  用户信息获取：
    - `/auth/users/me` 端点在有效令牌下返回当前用户信息。
    - 在未认证或令牌无效时拒绝访问。
4.  密码安全：
    - 确保 API 响应中不泄露 `password_hash`。
5.  WebSocket 连接认证：
    - 测试 `get_current_user_from_websocket_header` 依赖。
    - 通过 HTTP `Authorization` 头部成功认证 WebSocket 连接。
    - 处理无效/过期 Token、缺少或格式错误的认证头部、用户不存在等情况。
6.  测试环境与数据隔离：
    - 使用内存 SQLite 数据库进行独立的、可重复的测试。
    - 通过 PyTest fixtures 管理数据库连接和依赖覆盖。

通过这些自动化测试，确保认证系统的健壮性、安全性和正确性，
并为后续的代码重构和功能迭代提供信心。
"""
import pytest
import sqlite3
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock # unittest.mock for synchronous and async
from datetime import datetime, timedelta, timezone # Ensure timezone is imported
from jose import jwt # For creating malformed tokens for testing

# 假设这些模块和应用实例可以从项目结构中正确导入
# 请根据您的项目结构调整导入路径
from app.backend.main import app  # FastAPI app instance
from app.backend.auth import schemas, security, crud # Assuming __init__.py exports these
# from app.backend.auth.models import authenticate_token # Removed as the function is removed
from app.backend.auth.deps import get_current_user_from_websocket_header # Updated import
from app.backend.auth.db import open_db_connection # DB_FILE, DATA_DIR might not be needed directly if overriding

# 用于测试的客户端
client = TestClient(app)

# --- 测试数据库设置 ---
TEST_DB_FILE = ":memory:" # 使用内存数据库进行测试

# Global test user details
TEST_USER_USERNAME = "testuser"
TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword"

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    # This fixture can be used for session-wide setup if needed.
    # For overriding dependencies, it's often done directly or via module-scoped fixtures.
    pass

@pytest.fixture
def db(): # Renamed from override_get_db for pytest convention
    # This fixture provides a fresh in-memory database for each test function that uses it.
    # It ensures tables are created.
    conn = sqlite3.connect(TEST_DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        role TEXT DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    """)
    # User_documents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        document_id TEXT NOT NULL,
        custom_filename TEXT,
        is_private BOOLEAN DEFAULT TRUE,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)
    conn.commit()
    
    # Override the app's dependency for open_db_connection
    # This needs to be a function that FastAPI can call to get the dependency
    def get_test_db_connection():
        # For TestClient, the connection needs to be yielded if it's managed by a context manager
        # However, since we are managing it per test with this fixture, we can just return it.
        # But FastAPI expects a generator for `yield` based dependencies.
        # For simplicity in testing individual CRUD or model functions that take `db` as an arg,
        # this fixture can just return `conn`.
        # For API tests using TestClient, the dependency_override is more robust.
        return conn 

    original_open_db_connection = app.dependency_overrides.get(open_db_connection)
    app.dependency_overrides[open_db_connection] = get_test_db_connection
    
    yield conn # Provide the connection to the test

    # Teardown: close connection and clear override
    conn.close()
    if original_open_db_connection:
        app.dependency_overrides[open_db_connection] = original_open_db_connection
    else:
        del app.dependency_overrides[open_db_connection]


# --- 辅助函数 ---
def create_test_user_in_db(db_conn: sqlite3.Connection, user_data: schemas.UserCreate) -> schemas.UserInDB:
    # Ensure the user is created for testing purposes
    # This helper assumes the 'users' table exists, created by the 'db' fixture
    existing_user = crud.get_user_by_username(db_conn, user_data.username)
    if existing_user:
        return existing_user # Return existing user if already created by a previous step/test

    # If not existing, create new
    created_user_internal = crud.create_user(db=db_conn, user=user_data)
    if not created_user_internal:
        raise ValueError("Failed to create test user in DB for testing")
    return created_user_internal


# --- 测试登录逻辑与 password_hash 处理 (API Tests) ---

def test_user_registration_success(db): # db fixture will set up the in-memory DB
    response = client.post(
        "/auth/register",
        json={"username": "newapiuser", "email": "newapiuser@example.com", "password": "newapipassword", "role": "user"},
    )
    assert response.status_code == 200, response.text
    user_data = response.json()
    assert user_data["username"] == "newapiuser"
    assert user_data["email"] == "newapiuser@example.com"
    assert "id" in user_data
    assert "password_hash" not in user_data # 关键验证
    assert user_data["role"] == "user"

def test_user_registration_duplicate_email(db):
    client.post(
        "/auth/register",
        json={"username": "anotherapiuser", "email": "apiduplicate@example.com", "password": "password1", "role": "user"},
    )
    response = client.post(
        "/auth/register",
        json={"username": "anotherapiuser2", "email": "apiduplicate@example.com", "password": "password2", "role": "user"},
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Email already registered"

def test_user_registration_duplicate_username(db):
    client.post(
        "/auth/register",
        json={"username": "apiduplicateuser", "email": "api_unique_email@example.com", "password": "password1", "role": "user"},
    )
    response = client.post(
        "/auth/register",
        json={"username": "apiduplicateuser", "email": "api_unique_email2@example.com", "password": "password2", "role": "user"},
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Username already registered"

def test_user_login_success(db):
    create_test_user_in_db(db, schemas.UserCreate(username=TEST_USER_USERNAME, email=TEST_USER_EMAIL, password=TEST_USER_PASSWORD, role="user"))
    response = client.post(
        "/auth/login/token",
        data={"username": TEST_USER_USERNAME, "password": TEST_USER_PASSWORD},
    )
    assert response.status_code == 200, response.text
    token_data = response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"

def test_user_login_wrong_password(db):
    create_test_user_in_db(db, schemas.UserCreate(username="loginfailuser_api", email="loginfail_api@example.com", password="correctpassword", role="user"))
    response = client.post(
        "/auth/login/token",
        data={"username": "loginfailuser_api", "password": "wrongpassword"},
    )
    assert response.status_code == 401, response.text
    assert response.json()["detail"] == "Incorrect username or password"

def test_user_login_user_not_found(db):
    response = client.post(
        "/auth/login/token",
        data={"username": "nonexistentuser_api", "password": "anypassword"},
    )
    assert response.status_code == 401, response.text
    assert response.json()["detail"] == "Incorrect username or password"

def test_read_users_me_success(db):
    # Register and login user
    reg_response = client.post(
        "/auth/register",
        json={"username": "me_user_api", "email": "me_api@example.com", "password": "mepassword", "role": "user"},
    )
    assert reg_response.status_code == 200, reg_response.text
    
    login_response = client.post(
        "/auth/login/token",
        data={"username": "me_user_api", "password": "mepassword"},
    )
    assert login_response.status_code == 200, login_response.text
    access_token = login_response.json()["access_token"]

    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/auth/users/me", headers=headers)
    
    assert response.status_code == 200, response.text
    user_data = response.json()
    assert user_data["username"] == "me_user_api"
    assert user_data["email"] == "me_api@example.com"
    assert "password_hash" not in user_data

def test_read_users_me_unauthenticated(db):
    response = client.get("/auth/users/me")
    assert response.status_code == 401, response.text # oauth2_scheme auto_error is True by default in get_current_user
    # The detail might be "Not authenticated" if token is missing, or "Could not validate credentials" if present but invalid
    # For deps.get_current_user, if no token, it raises credentials_exception.
    assert response.json()["detail"] == "Could not validate credentials"


# --- 测试 get_current_user_from_websocket_header (deps.py) --- # Updated section title

@pytest.fixture
def test_user_for_websocket(db):
    user_create = schemas.UserCreate(username="ws_user", email="wsuser@example.com", password="wspassword", role="user")
    return create_test_user_in_db(db, user_create)

@pytest.mark.asyncio
async def test_get_current_user_from_websocket_success(db, test_user_for_websocket):
    token = security.create_access_token(data={"sub": test_user_for_websocket.username})
    
    mock_websocket = AsyncMock()
    # Simulate headers in ASGI scope
    mock_websocket.scope = {
        "type": "websocket",
        "headers": [(b"authorization", f"Bearer {token}".encode("utf-8"))]
    }
    mock_websocket.url = "ws://localhost/ws" # URL doesn't need token anymore

    user = await get_current_user_from_websocket_header(websocket=mock_websocket, db=db) # Use updated function
    
    assert user is not None
    assert isinstance(user, schemas.UserInDB)
    assert user.username == test_user_for_websocket.username

@pytest.mark.asyncio
async def test_get_current_user_from_websocket_invalid_token(db):
    mock_websocket = AsyncMock()
    mock_websocket.scope = {
        "type": "websocket",
        "headers": [(b"authorization", b"Bearer invalid.jwt.token")]
    }
    mock_websocket.url = "ws://localhost/ws"
    user = await get_current_user_from_websocket_header(websocket=mock_websocket, db=db)
    assert user is None

@pytest.mark.asyncio
async def test_get_current_user_from_websocket_expired_token(db, test_user_for_websocket):
    expired_token = security.create_access_token(
        data={"sub": test_user_for_websocket.username},
        expires_delta=timedelta(seconds=-3600)
    )
    mock_websocket = AsyncMock()
    mock_websocket.scope = {
        "type": "websocket",
        "headers": [(b"authorization", f"Bearer {expired_token}".encode("utf-8"))]
    }
    mock_websocket.url = "ws://localhost/ws"
    user = await get_current_user_from_websocket_header(websocket=mock_websocket, db=db)
    assert user is None

@pytest.mark.asyncio
async def test_get_current_user_from_websocket_no_auth_header(db): # Test for missing header
    mock_websocket = AsyncMock()
    mock_websocket.scope = {"type": "websocket", "headers": []} # No Authorization header
    mock_websocket.url = "ws://localhost/ws"
    user = await get_current_user_from_websocket_header(websocket=mock_websocket, db=db)
    assert user is None

@pytest.mark.asyncio
async def test_get_current_user_from_websocket_malformed_auth_header(db): # Test for malformed header
    mock_websocket = AsyncMock()
    mock_websocket.scope = {
        "type": "websocket",
        "headers": [(b"authorization", b"Token someothertoken")] # Not "Bearer"
    }
    mock_websocket.url = "ws://localhost/ws"
    user = await get_current_user_from_websocket_header(websocket=mock_websocket, db=db)
    assert user is None

@pytest.mark.asyncio
async def test_get_current_user_from_websocket_user_not_in_db(db):
    cursor = db.cursor()
    cursor.execute("DELETE FROM users WHERE username = 'ghost_ws_user'") # Ensure not present
    db.commit()

    token_for_ghost = security.create_access_token(data={"sub": "ghost_ws_user"})
    mock_websocket = AsyncMock()
    mock_websocket.scope = {
        "type": "websocket",
        "headers": [(b"authorization", f"Bearer {token_for_ghost}".encode("utf-8"))]
    }
    mock_websocket.url = "ws://localhost/ws"
    user = await get_current_user_from_websocket_header(websocket=mock_websocket, db=db)
    assert user is None