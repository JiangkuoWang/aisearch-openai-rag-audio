import logging
import os
from pathlib import Path
import json
import numpy as np
import openai
import sys
from typing import Any, List, Callable, Optional, Union, Dict
import asyncio
import tempfile
import shutil # Added for directory cleanup new_pro

from aiohttp import web
import aiohttp_cors  # 添加CORS支持
from dotenv import load_dotenv

# Local imports
from rtmt import RTMiddleTier
from auth.router import router as auth_router
from rag_providers.base import BaseRAGProvider
from rag_providers.in_memory import InMemoryRAGProvider
from rag_providers.llama_index_graph import LlamaIndexGraphRAGProvider
from ragtools import attach_rag_tools
# Ensure rag_upload_utils.py exists in the same directory
try:
    from rag_upload_utils import extract_text, chunk_text
except ImportError:
    logging.error("Could not import from rag_upload_utils.py. Make sure the file exists and is correct.")
    # Define dummy functions to prevent crashing if import fails
    def extract_text(filename: str, raw: bytes) -> str: return ""
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]: return []

from auth_middleware import setup_auth_middleware
from auth.models import associate_document_with_user, get_user_document_ids
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicerag")

# Define backend directory for constructing absolute paths
BACKEND_DIR = Path(__file__).parent.resolve()

# --- Helper Function to Update RAG Provider ---
def update_rag_provider(app: web.Application, rag_provider: Optional[BaseRAGProvider]):
    """Dynamically attach/detach RAG tools based on the provider."""
    rtmt = app.get("rtmt")
    if not rtmt:
        logger.error("RTMiddleTier instance not found in app context.")
        return

    # Remove existing RAG tools first
    for name in ("search", "report_grounding"):
        if name in rtmt.tools:
            del rtmt.tools[name]
            logger.debug(f"Removed RAG tool: {name}")

    # Attach new tools if a provider is given
    if rag_provider:
        try:
            attach_rag_tools(rtmt, rag_provider)
            logger.info(f"Attached RAG tools using provider: {type(rag_provider).__name__}")
        except Exception as e:
            logger.exception(f"Error attaching RAG tools: {e}")
    else:
        logger.info("Detached RAG tools (provider is None).")

    app["rag_provider"] = rag_provider # Store the current provider

# --- HTTP Handlers ---
async def handle_rag_config(request: web.Request):
    """Handles POST request to set the RAG provider type."""
    try:
        data = await request.json()
        provider_type = data.get("provider_type", "").lower()
        if provider_type not in ("none", "in_memory", "llama_index"):
            logger.warning(f"Invalid provider_type received: {provider_type}")
            return web.HTTPBadRequest(text="Invalid provider_type specified. Must be 'none', 'in_memory', or 'llama_index'.")

        app = request.app
        app["rag_provider_type"] = provider_type
        logger.info(f"RAG provider type set to: {provider_type}")

        # 添加：获取当前用户ID（如果有）
        user_id = request.get("user_id")
        if user_id:
            logger.info(f"用户 {user_id} 设置RAG提供程序类型为: {provider_type}")
            # 将用户ID保存到应用上下文中，以便后续上传处理
            app["current_user_id"] = user_id
        
        # If 'none', detach tools immediately.
        if provider_type == "none":
            update_rag_provider(app, None)

        return web.json_response({
            "status": "ok", 
            "message": f"RAG provider type set to {provider_type}",
            "user_authenticated": user_id is not None
        })
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON in /rag-config request.")
        return web.HTTPBadRequest(text="Invalid JSON payload.")
    except Exception as e:
        logger.exception(f"Error handling RAG config: {e}")
        return web.HTTPInternalServerError(text="Error processing RAG configuration.")

async def handle_upload(request: web.Request):
    """Handles file uploads to build and activate a RAG provider."""
    app = request.app
    provider_type = app.get("rag_provider_type")

    if provider_type not in ("in_memory", "llama_index"):
        logger.warning(f"Upload attempt received but RAG type is '{provider_type}'. Ignoring.")
        return web.HTTPBadRequest(text=f"File upload not supported for RAG type '{provider_type}'. Select 'in_memory' or 'llama_index' first via /rag-config.")

    # Create a temporary directory for uploaded files
    temp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
    logger.info(f"Created temporary directory for uploads: {temp_dir}")
    try:
        reader = await request.multipart()
        texts: List[str] = []
        titles: List[str] = []
        file_paths: List[Path] = [] # Store paths for LlamaIndex

        part_index = 0
        while True:
            part = await reader.next()
            if part is None:
                break # End of parts
            if part.filename:
                # Sanitize filename if needed, here we just use it
                safe_filename = f"{part_index}_{Path(part.filename).name}"
                temp_file_path = temp_dir / safe_filename
                logger.info(f"Processing uploaded file: {part.filename} -> {temp_file_path}")
                file_size = 0
                try:
                    with open(temp_file_path, "wb") as f:
                        while True:
                            chunk = await part.read_chunk(size=8192) # Read in chunks
                            if not chunk:
                                break
                            f.write(chunk)
                            file_size += len(chunk)
                    logger.info(f"Saved {part.filename} ({file_size} bytes) to {temp_file_path}")
                    file_paths.append(temp_file_path) # Store path for later use

                    # Extract and chunk text immediately after saving
                    raw_data = temp_file_path.read_bytes()
                    try:
                        extracted_text = extract_text(part.filename, raw_data)
                        if not extracted_text:
                             logger.warning(f"No text extracted from {part.filename}, skipping.")
                             continue # Skip this file
                             
                        chunks = chunk_text(extracted_text) # Use default chunk size
                        if not chunks:
                            logger.warning(f"No chunks created from {part.filename}, skipping.")
                            continue # Skip this file
                            
                        logger.info(f"Extracted and chunked {part.filename} into {len(chunks)} chunks.")

                        # Append chunks and their titles
                        texts.extend(chunks)
                        titles.extend([part.filename] * len(chunks))
                    except ImportError as e:
                        # 明确捕获缺少依赖库的错误
                        logger.error(f"Missing required library for processing {part.filename}: {e}")
                        return web.HTTPInternalServerError(text=f"Missing required library for processing {part.filename}. Please install all required dependencies.")
                    except Exception as e:
                        logger.exception(f"Error extracting text from {part.filename}: {e}")
                        # 继续处理其他文件，不中断上传过程

                except Exception as e:
                    logger.exception(f"Error processing/saving file {part.filename}: {e}")
                    # Decide if one failed file should abort the whole upload
                    # return web.HTTPInternalServerError(text=f"Error processing file {part.filename}")
                part_index += 1
            else:
                # Handle non-file parts if necessary, e.g., form fields
                 logger.warning("Received a multipart part without a filename, ignoring.")


        if not texts:
            logger.error("No text could be extracted or chunked from uploaded files.")
            # Clean up temp dir as nothing was processed
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up empty temporary directory: {temp_dir}")
            return web.HTTPBadRequest(text="No processable content found in uploaded files.")

        logger.info(f"Total text chunks to process: {len(texts)}")

        # Get necessary components from app context
        openai_client = app.get("openai_client")
        embedding_model = app.get("openai_embedding_model")
        if not openai_client or not embedding_model:
             logger.error("OpenAI client or embedding model not found in app context.")
             return web.HTTPInternalServerError(text="Backend configuration error.")

        logger.info(f"Generating embeddings using model: {embedding_model}...")
        try:
            # Ensure client is async if using await
            if not isinstance(openai_client, openai.AsyncOpenAI):
                 logger.error("OpenAI client is not async, cannot await.")
                 # Fallback or error - assuming sync client for now if needed, but should be async
                 # For demo, let's assume it *is* async as intended
                 return web.HTTPInternalServerError(text="Internal configuration error (OpenAI client type).")

            # Batch embedding generation
            response = await openai_client.embeddings.create(input=texts, model=embedding_model)
            vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
            logger.info(f"Generated {vectors.shape[0]} embeddings with dimension {vectors.shape[1]}.")
        except Exception as e:
             logger.exception("Error generating embeddings.")
             return web.HTTPInternalServerError(text="Failed to generate embeddings for uploaded content.")

        # Prepare metadata list matching the order of texts/vectors
        metadata_list = [
            {"chunk_id": f"{titles[i]}-{i}", "text": texts[i], "title": titles[i]}
            for i in range(len(texts))
        ]

        new_provider: Optional[BaseRAGProvider] = None
        if provider_type == "in_memory":
            logger.info("Initializing InMemoryRAGProvider...")
            try:
                new_provider = InMemoryRAGProvider(
                    openai_client=openai_client, # Pass the client object
                    embedding_model=embedding_model,
                    all_metadata=metadata_list,
                    all_vectors=vectors
                )
                logger.info("InMemoryRAGProvider initialized.")
            except Exception as e:
                 logger.exception("Failed to initialize InMemoryRAGProvider")
                 return web.HTTPInternalServerError(text="Failed to initialize In-Memory RAG provider.")

        elif provider_type == "llama_index":
            from scripts.create_llama_graph_index import create_graph_index
            llama_index_persist_dir = temp_dir / "llama_index_data"
            llama_index_persist_dir.mkdir(exist_ok=True)    
            index = await asyncio.to_thread(
                create_graph_index,
                source_dir=str(temp_dir),
                index_dir=str(llama_index_persist_dir)
            )
           
            # 创建检索器
            retriever = index.as_retriever(similarity_top_k=5)
            # 实例化并设置provider
            new_provider = LlamaIndexGraphRAGProvider(
                openai_client=openai_client,
                index_dir=llama_index_persist_dir,
                embedding_model_name=embedding_model,
                llm_model_name=app["openai_model"],
                    )
                    
            # 设置索引和检索器
            new_provider.index = index
            new_provider.retriever = retriever
            logger.info("LlamaIndexGraphRAGProvider已初始化，配置了PropertyGraphIndex和检索器")
            

        # Activate the new provider
        if new_provider:
            update_rag_provider(app, new_provider)
            logger.info(f"Successfully activated {provider_type} RAG provider with uploaded data.")
            
            # 添加：如果请求中有用户信息，将文档与用户关联
            user_id = request.get("user_id")
            if user_id:
                db_conn_upload: Optional[sqlite3.Connection] = None
                try:
                    # 获取数据库连接
                    from auth.db import open_db_connection # 使用新的函数
                    db_conn_upload = open_db_connection() # 直接获取连接
                    
                    # 关联每个文件路径与用户
                    for i, file_path in enumerate(file_paths):
                        # 使用文件路径作为document_id
                        document_id = f"{provider_type}_{file_path.name}"
                        custom_filename = Path(file_path.name).name  # 使用原始文件名
                        
                        success = associate_document_with_user(
                            db_conn_upload, 
                            user_id, 
                            document_id, 
                            custom_filename=custom_filename
                        )
                        if success:
                            logger.info(f"文档 {document_id} 已关联到用户 {user_id}")
                    
                except Exception as e:
                    logger.exception(f"关联文档到用户时出错: {e}")
                    # 不要因为关联失败而中断上传过程
                finally:
                    if db_conn_upload:
                        db_conn_upload.close()
            
            # 清理临时目录代码保持不变
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
            return web.json_response({
                "status": "ok",
                "message": f"{provider_type} RAG provider activated with {len(texts)} chunks from {part_index} files."
            })
        else:
            logger.error("Failed to create a RAG provider instance after processing uploads.")
            # Clean up temp dir as provider creation failed
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory due to provider creation failure: {temp_dir}")
            return web.HTTPInternalServerError(text="Failed to create RAG provider instance.")

    except Exception as e:
        logger.exception(f"Unhandled error during file upload processing: {e}")
        # Ensure cleanup happens on any unexpected error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.warning(f"Cleaned up temporary directory due to unhandled error: {temp_dir}")
        return web.HTTPInternalServerError(text="An unexpected error occurred during file upload processing.")

async def handle_auth_status(request: web.Request):
    """返回当前用户的认证状态信息"""
    # 获取用户信息（如果已认证）
    user = request.get("user")
    user_id = request.get("user_id")
    
    if user:
        return web.json_response({
            "authenticated": True,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role
            }
        })
    else:
        return web.json_response({
            "authenticated": False,
            "auth_url": "http://127.0.0.1:8765/auth"  # 认证服务的基URL
        })

# --- Main Application Setup ---
async def create_app():
    app = web.Application()
    
    # 配置CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        )
    })
    
    # Load .env file if not in production
    if not os.environ.get("RUNNING_IN_PRODUCTION"):
        logger.info("Running in development mode, loading from .env file")
        env_path = BACKEND_DIR / ".env"
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path)
            logger.info(f"Loaded environment variables from: {env_path}")
        else:
            logger.warning(f".env file not found at {env_path}, relying on existing environment variables.")

    # --- Load OpenAI Configuration ---
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview") # Ensure this is a valid model
    openai_embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large") # Ensure this is valid

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # --- Initialize OpenAI Async Client and store in app context ---
    # Use Async Client for await operations like embeddings
    try:
        openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        app["openai_client"] = openai_client
        app["openai_model"] = openai_model
        app["openai_embedding_model"] = openai_embedding_model
        logger.info("AsyncOpenAI client initialized.")
    except Exception as e:
        logger.exception("Failed to initialize AsyncOpenAI client.")
        raise ValueError(f"Could not initialize OpenAI client: {e}")


    # --- RAG Configuration State --- Initialized Dynamically
    app["rag_provider_type"] = "none" # Default to no RAG
    app["rag_provider"] = None        # No provider active initially

    # --- Initialize RTMiddleTier and store in app context ---
    rtmt = RTMiddleTier(
        openai_api_key=openai_api_key,
        model=openai_model,
        voice_choice=os.environ.get("OPENAI_REALTIME_VOICE_CHOICE") or "alloy"
    )
    app["rtmt"] = rtmt

    # Configure System Prompt for RTMiddleTier
    # Ensure this prompt reflects the dynamic RAG capability if needed
    rtmt.system_message = """
        You are a helpful assistant. If the user has enabled the knowledge base search, only answer questions based on information found using the 'search' tool. Otherwise, answer normally.
        The user is listening to answers with audio, so keep answers concise, ideally a single sentence.
        Never read file names, source names, or chunk IDs out loud.
        If using the knowledge base:
        1. Always use the 'search' tool first.
        2. Always use the 'report_grounding' tool to cite sources used.
        3. If the answer isn't in the knowledge base, state that.
    """.strip()
    logger.info("RTMiddleTier initialized with system prompt.")

    # --- Attach WebSocket and Static Routes ---
    # RAG tools are NOT attached here; they are attached dynamically by update_rag_provider
    rtmt.attach_to_app(app, "/realtime")
    logger.info("Attached WebSocket handler to /realtime")

    # 添加auth路由处理
    # 由于auth/router.py使用FastAPI定义的路由，我们需要为每个路由创建aiohttp处理程序
    # 注册端点
    async def handle_auth_register(request: web.Request) -> web.Response:
        db_conn: Optional[sqlite3.Connection] = None
        try:
            data = await request.json()
            # 获取数据库连接
            from auth.db import open_db_connection # 使用新的函数
            db_conn = open_db_connection() # 直接获取连接
            
            from auth import crud, schemas
            # 检查用户名和邮箱是否已存在
            db_user_by_email = crud.get_user_by_email(db_conn, email=data.get("email"))
            if db_user_by_email:
                return web.json_response(
                    {"detail": "Email already registered"},
                    status=400
                )
            
            db_user_by_username = crud.get_user_by_username(db_conn, username=data.get("username"))
            if db_user_by_username:
                return web.json_response(
                    {"detail": "Username already registered"},
                    status=400
                )
            
            # 创建用户
            user_create = schemas.UserCreate(
                username=data.get("username"),
                email=data.get("email"),
                password=data.get("password")
            )
            created_user = crud.create_user(db=db_conn, user=user_create)
            if not created_user:
                return web.json_response(
                    {"detail": "Could not create user"},
                    status=500
                )
            
            # 返回用户信息（不包括密码）
            return web.json_response({
                "id": created_user.id,
                "username": created_user.username,
                "email": created_user.email,
                "is_active": created_user.is_active,
                "role": created_user.role,
                "created_at": created_user.created_at.isoformat() if created_user.created_at else None
            })
        except Exception as e:
            logger.exception(f"注册用户时出错: {e}")
            return web.json_response(
                {"detail": f"Registration error: {str(e)}"},
                status=500
            )
        finally:
            if db_conn:
                db_conn.close()
    
    # 登录端点
    async def handle_auth_login(request: web.Request) -> web.Response:
        db_conn: Optional[sqlite3.Connection] = None
        try:
            data = await request.json()
            username = data.get("username")
            password = data.get("password")
            
            # 获取数据库连接
            from auth.db import open_db_connection # 使用新的函数
            db_conn = open_db_connection() # 直接获取连接
            
            from auth import crud, security
            # 验证用户
            user = crud.get_user_by_username(db_conn, username=username)
            if not user or not security.verify_password(password, user.password_hash):
                return web.json_response(
                    {"detail": "Incorrect username or password"},
                    status=401
                )
            
            # 创建访问令牌
            from datetime import timedelta
            access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = security.create_access_token(
                data={"sub": user.username}, expires_delta=access_token_expires
            )
            
            # 返回用户信息
            return web.json_response({
                "access_token": access_token,
                "token_type": "bearer",
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "role": user.role
                }
            })
        except Exception as e:
            logger.exception(f"用户登录时出错: {e}")
            return web.json_response(
                {"detail": f"Login error: {str(e)}"},
                status=500
            )
        finally:
            if db_conn:
                db_conn.close()
    
    # 添加auth路由
    auth_register = app.router.add_post("/auth/register", handle_auth_register)
    auth_login = app.router.add_post("/auth/login", handle_auth_login)
    
    # 将auth路由添加到CORS配置
    cors.add(auth_register)
    cors.add(auth_login)
    
    logger.info("添加了认证路由: /auth/register, /auth/login")

    # Add HTTP routes for RAG config and upload
    rag_config_resource = app.router.add_post("/rag-config", handle_rag_config)
    upload_resource = app.router.add_post("/upload", handle_upload)
    auth_status_resource = app.router.add_get("/auth-status", handle_auth_status)
    
    # 将路由添加到CORS配置
    cors.add(rag_config_resource)
    cors.add(upload_resource)
    cors.add(auth_status_resource)
    
    logger.info("Added HTTP routes: /rag-config (POST), /upload (POST), /auth-status (GET)")


    # Serve static frontend files (assuming frontend build places files in backend/static)
    static_dir = BACKEND_DIR / 'static'
    if not static_dir.exists() or not (static_dir / 'index.html').exists():
        logger.warning(f"Static directory '{static_dir}' or index.html not found. Frontend may not be served.")
    else:
        logger.info(f"Serving static files from: {static_dir}")
        # Serve index.html at the root
        app.router.add_get('/', lambda _: web.FileResponse(static_dir / 'index.html'))
        # Serve other static files (JS, CSS, assets)
        app.router.add_static('/', path=static_dir, name='static', show_index=False) # Important: show_index=False

    # 添加：设置认证中间件
    try:
        setup_auth_middleware(app)
        logger.info("已设置认证中间件")
    except Exception as e:
        logger.exception(f"设置认证中间件时出错: {e}")
    
    return app

# --- Application Entry Point ---
if __name__ == "__main__":
    # Define an async main function to create the app instance
    async def main():
        # Setup logging levels based on environment or defaults
        # logging.getLogger('aiohttp.access').setLevel(logging.WARNING) # Example: reduce access log noise
        return await create_app()

    host = os.environ.get("BACKEND_HOST", "127.0.0.1") # Default to 127.0.0.1 for local dev
    port = int(os.environ.get("BACKEND_PORT", 8765)) # Use a distinct port

    logger.info(f"Starting application server on http://{host}:{port}")

    # Run the app using aiohttp's web runner
    # The 'main()' function returns the app instance created by create_app()
    web.run_app(main(), host=host, port=port)
