import sqlite3
import os

from app.backend.config import config_service # Added
from pathlib import Path # Added

# 定义数据库文件的名字和路径
# 从 config_service 获取数据库路径
DB_FILE_PATH = config_service.settings.DATABASE_PATH.resolve()


def create_connection(db_file_path: Path): # Changed type hint
    """ 创建一个数据库连接到SQLite数据库 """
    conn = None
    try:
        # 如果目录不存在，则创建它
        os.makedirs(os.path.dirname(db_file_path), exist_ok=True)
        conn = sqlite3.connect(db_file_path)
        print(f"SQLite version: {sqlite3.sqlite_version}")
        print(f"Successfully connected to database: {db_file_path}")
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
    return conn

def create_table(conn, create_table_sql):
    """ 使用提供的SQL语句创建表 """
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit() # 确保 DDL 操作被提交
        print(f"Table created successfully (or already exists): {create_table_sql.split('(')[0].split('EXISTS')[-1].strip()}")
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")

def main():
    # SQL 语句用于创建表
    sql_create_users_table = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_active INTEGER DEFAULT 0, -- 0 for inactive, 1 for active
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        last_login TEXT,
        role TEXT DEFAULT 'user' NOT NULL
    );
    """

    sql_create_user_settings_table = """
    CREATE TABLE IF NOT EXISTS user_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        preference_key TEXT NOT NULL,
        preference_value TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE (user_id, preference_key)
    );
    """

    sql_create_social_accounts_table = """
    CREATE TABLE IF NOT EXISTS social_accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        provider_type TEXT NOT NULL, -- e.g., 'google', 'microsoft'
        external_user_id TEXT NOT NULL,
        provider_data TEXT, -- Store as JSON string
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE (provider_type, external_user_id)
    );
    """

    sql_create_user_session_history_table = """
    CREATE TABLE IF NOT EXISTS user_session_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id TEXT UNIQUE NOT NULL,
        started_at TEXT DEFAULT CURRENT_TIMESTAMP,
        ended_at TEXT,
        ip_address TEXT,
        user_agent TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """

    sql_create_user_documents_table = """
    CREATE TABLE IF NOT EXISTS user_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        document_id TEXT NOT NULL,
        upload_time TEXT DEFAULT CURRENT_TIMESTAMP,
        is_private INTEGER DEFAULT 1, -- 1 for private, 0 for public/shared
        custom_filename TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE (user_id, document_id)
    );
    """

    # --- (可选) 如果你决定有一个单独的 documents 表 ---
    # sql_create_documents_table = """
    # CREATE TABLE IF NOT EXISTS documents (
    #     id TEXT PRIMARY KEY, -- e.g., a UUID
    #     file_path TEXT NOT NULL,
    #     file_type TEXT,
    #     file_size INTEGER,
    #     created_at TEXT DEFAULT CURRENT_TIMESTAMP
    # );
    # """

    # 创建数据库连接
    conn = create_connection(DB_FILE_PATH) # Use DB_FILE_PATH from config

    if conn is not None:
        # 创建表
        create_table(conn, sql_create_users_table)
        create_table(conn, sql_create_user_settings_table)
        create_table(conn, sql_create_social_accounts_table)
        create_table(conn, sql_create_user_session_history_table)
        create_table(conn, sql_create_user_documents_table)
        # create_table(conn, sql_create_documents_table) # 如果使用独立的 documents 表，取消这行注释

        # (可选) 创建索引的示例
        try:
            cursor = conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_social_accounts_external_user_id ON social_accounts(external_user_id);")
            conn.commit()
            print("Indexes created successfully or already exist.")
        except sqlite3.Error as e:
            print(f"Error creating indexes: {e}")


        # 关闭连接
        conn.close()
        print(f"Database setup complete. Database file at: {DB_FILE_PATH}") # Use DB_FILE_PATH
    else:
        print("Error! Cannot create the database connection.")

if __name__ == '__main__':
    main()