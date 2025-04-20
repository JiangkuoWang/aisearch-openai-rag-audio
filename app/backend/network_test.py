# test_openai_ws.py (放在项目根目录或 backend 目录)
import websocket
import os
import json
import logging  # 添加 logging
from dotenv import load_dotenv

# 配置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 根据脚本位置动态调整 .env 文件路径
# 检查 'app/backend/.env' 是否存在，如果不存在则使用根目录的 '.env'
dotenv_path = 'app/backend/.env' if os.path.exists('app/backend/.env') else '.env'
load_dotenv(dotenv_path=dotenv_path)
logging.info(f"从 {dotenv_path} 加载 .env 文件")

# 从环境变量获取 OpenAI API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# 从环境变量获取模型 ID，如果未设置，则使用默认值 "gpt-4o-realtime-preview"
MODEL_ID = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")

# 检查 API Key 是否存在
if not OPENAI_API_KEY:
    logging.error("错误: 未在环境变量中找到 OPENAI_API_KEY")
    exit(1) # 退出程序，返回非零状态码表示错误

# 构建 WebSocket 连接 URL
URL = f"wss://api.openai.com/v1/realtime?model={MODEL_ID}"
# 设置请求头
HEADERS = [
    f"Authorization: Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta: realtime=v1"
]

logging.info(f"尝试连接到 WebSocket: {URL}")
logging.info(f"使用的 Headers: {HEADERS}") # 打印 headers 以供调试

def on_open(ws):
    """WebSocket 连接成功打开时的回调函数"""
    logging.info("WebSocket 连接已成功打开")
    # 连接成功后可以立即关闭进行简单的连接测试
    # ws.close() # 如果只想测试连接，取消此行注释

def on_message(ws, message):
    """接收到 WebSocket 消息时的回调函数"""
    logging.info(f"收到消息: {message}")
    try:
        # 尝试解析 JSON 消息
        data = json.loads(message)
        # 在这里处理接收到的数据
        # print(f"解析后的数据: {data}")
    except json.JSONDecodeError:
        logging.warning(f"收到的消息不是有效的 JSON 格式: {message}")

def on_error(ws, error):
    """WebSocket 连接发生错误时的回调函数"""
    logging.error(f"发生错误: {error}")

def on_close(ws, close_status_code, close_msg):
    """WebSocket 连接关闭时的回调函数"""
    logging.info(f"WebSocket 连接已关闭: 状态码={close_status_code}, 原因={close_msg}")

# --- WebSocket 应用设置 ---
# 启用详细跟踪，方便查看 WebSocket 握手和通信过程 (调试时取消注释)
# websocket.enableTrace(True)

# 创建 WebSocketApp 实例
ws_app = websocket.WebSocketApp(
    URL,
    header=HEADERS,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
)

# --- 运行 WebSocket 客户端 ---
# 运行 WebSocket 客户端，保持长连接
# ping_timeout 设置心跳超时时间（秒），防止连接因不活跃而断开
# sockopt 可以用来设置底层的 socket 选项，例如 TCP 超时 (需要 websocket-client 0.57.0+)
# 示例：设置 TCP 连接超时 (需要导入 socket 模块)
# import socket
# ws_app.run_forever(
#     sockopt=((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),), # 禁用 Nagle 算法
#     ping_interval=10, # 每 10 秒发送一次 ping
#     ping_timeout=5    # 5 秒内未收到 pong 则认为超时
# )
logging.info("启动 WebSocket 客户端...")
# 使用默认设置运行，添加 ping_timeout 保持连接
ws_app.run_forever(ping_timeout=20)

logging.info("WebSocket 客户端已停止。")
