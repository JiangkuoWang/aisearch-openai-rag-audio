import sqlite3
import os

# 修改：使用与setup_database.py相同的路径计算逻辑
# setup_database.py中使用的是：
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # app目录的上一级，即项目根目录
# DATA_DIR = os.path.join(BASE_DIR, 'data')

# 现在在auth/db.py中：
# __file__是app/backend/auth/db.py的路径
# 向上3级即为项目根目录（aisearch-openai-rag-audio）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_FILE = os.path.join(DATA_DIR, 'voice_rag_auth.db')

# 调试输出，确认路径一致性
print(f"Database location (from db.py): {DB_FILE}")

def open_db_connection() -> sqlite3.Connection: # 重命名以明确其行为
    """
    打开并返回一个新的数据库连接。
    调用者负责关闭此连接。
    """
    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        # 添加 check_same_thread=False 参数，允许在不同线程使用同一连接
        # 注意：这种方式在高并发环境下可能会有同步问题，但对于测试和中小型应用足够了
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        # 配置 row_factory 以便将行作为字典访问 (或者 sqlite3.Row 对象)
        # conn.row_factory = sqlite3.Row 
        # 不过我们的 CRUD 操作目前是手动将元组转换为字典，所以这里不是严格必需
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        # 在实际应用中，这里可能需要更复杂的错误处理或日志记录
        # 如果连接失败，FastAPI 通常会返回 500 错误
        raise # 重新引发异常，以便 FastAPI 可以捕获它
    # finally 块被移除，连接不再在此处关闭 