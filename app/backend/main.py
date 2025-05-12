import logging
import os
from pathlib import Path
import json # Added
import tempfile # Added
import shutil # Added
import asyncio # Added
from typing import Optional, List, Dict, Any # List, Dict, Any Added

import openai
import uvicorn
import numpy as np # Added
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile, WebSocket, WebSocketDisconnect, status # Request, File, UploadFile, Depends Added. WebSocket, WebSocketDisconnect, status Added
from fastapi.responses import JSONResponse, FileResponse # Added FileResponse
from pydantic import BaseModel # Added
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# Local imports
from app.backend.auth.router import router as auth_router
from app.backend.auth.deps import get_current_user, get_current_user_or_none, get_current_user_from_websocket_query_param # Added get_current_user_from_websocket_query_param
from app.backend.auth.schemas import UserInDB
from app.backend.auth.models import associate_document_with_user
from app.backend.auth.db import open_db_connection # Added
# Assuming RTMiddleTier and BaseRAGProvider are in the same directory or accessible via Python path
# For BaseRAGProvider, it's used as a type hint in update_rag_provider
try:
    from .rtmt import RTMiddleTier
    from .rag_providers.base import BaseRAGProvider
    from .rag_providers.in_memory import InMemoryRAGProvider # Added
    from .rag_providers.llama_index_graph import LlamaIndexGraphRAGProvider # Added
    from .ragtools import attach_rag_tools
    from .rag_upload_utils import extract_text, chunk_text # Added
except ImportError as e:
    print(f"Error importing local modules: {e}. Ensure rtmt.py, rag_providers/*, ragtools.py, and rag_upload_utils.py are accessible.")
    # Define placeholders if running in an environment where these are not critical for initial setup
    class RTMiddleTier:
        def __init__(self, *args, **kwargs): pass
        tools: dict = {}
        system_message: str = ""

    class BaseRAGProvider: pass
    class InMemoryRAGProvider(BaseRAGProvider): pass
    class LlamaIndexGraphRAGProvider(BaseRAGProvider): pass
    def attach_rag_tools(rtmt, provider): pass
    def extract_text(filename: str, raw: bytes) -> str: return ""
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]: return []


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicerag")

# --- Backend Directory ---
BACKEND_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BACKEND_DIR / "static"

# --- FastAPI Application Instance ---
app = FastAPI()

# --- Helper Function to Update RAG Provider (Migrated and Adapted) ---
def update_rag_provider(current_app: FastAPI, rag_provider: Optional[BaseRAGProvider]):
    """Dynamically attach/detach RAG tools based on the provider."""
    rtmt = getattr(current_app.state, "rtmt", None)
    if not rtmt:
        logger.error("RTMiddleTier instance not found in app.state.")
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

    current_app.state.rag_provider = rag_provider

# --- Application Startup Logic ---
@app.on_event("startup")
async def startup_event():
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
    openai_model_name = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
    openai_embedding_model_name = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # --- Initialize OpenAI Async Client and store in app.state ---
    try:
        openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        app.state.openai_client = openai_client
        app.state.openai_model = openai_model_name
        app.state.openai_embedding_model = openai_embedding_model_name
        logger.info("AsyncOpenAI client initialized and stored in app.state.")
    except Exception as e:
        logger.exception("Failed to initialize AsyncOpenAI client.")
        raise ValueError(f"Could not initialize OpenAI client: {e}")

    # --- Initialize RTMiddleTier and store in app.state ---
    try:
        rtmt_instance = RTMiddleTier(
            openai_api_key=openai_api_key,
            model=openai_model_name,
            voice_choice=os.environ.get("OPENAI_REALTIME_VOICE_CHOICE") or "alloy"
        )
        # Configure System Prompt for RTMiddleTier
        rtmt_instance.system_message = """
You are a helpful assistant. If the user has enabled the knowledge base search, only answer questions based on information found using the 'search' tool. Otherwise, answer normally.
The user is listening to answers with audio, so keep answers concise, ideally a single sentence.
Never read file names, source names, or chunk IDs out loud.
If using the knowledge base:
1. Always use the 'search' tool first.
2. Always use the 'report_grounding' tool to cite sources used.
        """.strip()
        app.state.rtmt = rtmt_instance
        logger.info("RTMiddleTier initialized and stored in app.state.")
    except Exception as e:
        logger.exception("Failed to initialize RTMiddleTier.")

    # --- RAG Configuration Initial State ---
    app.state.rag_provider_type = "none"
    app.state.rag_provider = None
    logger.info("Initial RAG provider state set to 'none'.")
    logger.info("FastAPI application startup sequence complete.") # Added log message

# --- CORS Configuration ---
origins = [
    "http://localhost:8765",
    "http://127.0.0.1:8765",
    "http://localhost:5173",  # 保留旧的前端开发服务器源以实现兼容性
]
# 检查环境变量是否覆盖
env_origins_str = os.environ.get("FRONTEND_URL")
if env_origins_str:
    logger.info(f"Overriding CORS origins with FRONTEND_URL: {env_origins_str}")
    origins = [origin.strip() for origin in env_origins_str.split(",")]

# 处理 "*" 通配符的特殊情况
if "*" in origins and len(origins) > 1:
    logger.warning("CORS allow_origins contains '*' along with specific origins. Using '*' only for maximum permissiveness as per configuration.")
    origins = ["*"]
elif not origins: # 如果环境变量为空或未设置，则使用默认值
    logger.warning("FRONTEND_URL was empty or not set, and no default origins were added. Falling back to restrictive defaults.")
    origins = [
        "http://localhost:8765", # 确保新的默认值包含这些
        "http://127.0.0.1:8765",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # 允许所有标准方法
    allow_headers=["*"], # 允许所有头部
)
logger.info(f"CORS middleware configured for origins: {origins}")

# --- Include Authentication Routes ---
app.include_router(auth_router, tags=["Authentication"])
logger.info("Authentication routes from app.backend.auth.router included.")

# --- Static File Routes and Root Path Configuration ---

# Serve index.html at the root path
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not STATIC_DIR.is_dir(): # Check STATIC_DIR existence
        logger.error(f"Static files directory not found: {STATIC_DIR}. Cannot serve index.html.")
        raise HTTPException(status_code=500, detail="Server configuration error: Static directory not found.")
    if not index_path.is_file():
        logger.error(f"Root index.html not found at {index_path}")
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

# Serve favicon.ico
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # STATIC_DIR existence is crucial for serving any static file.
    if not STATIC_DIR.is_dir():
        logger.error(f"Static files directory not found: {STATIC_DIR}. Cannot serve favicon.ico.")
        raise HTTPException(status_code=500, detail="Server configuration error: Static directory not found.")
    favicon_path = STATIC_DIR / "favicon.ico"
    if not favicon_path.is_file():
        logger.warning(f"favicon.ico not found at {favicon_path}") # Warning as it's less critical than index.html
        raise HTTPException(status_code=404, detail="favicon.ico not found")
    return FileResponse(favicon_path)

# General health check logs for static file serving (similar to original)
if not STATIC_DIR.is_dir():
    logger.warning(f"Primary static files directory ({STATIC_DIR}) not found. Core static assets (index.html, favicon.ico) might fail to load.")
elif not (STATIC_DIR / "index.html").is_file():
    logger.warning(f"index.html not found in {STATIC_DIR}. Root path '/' will not serve the frontend despite the route existing.")

# Note: app.mount for /assets will be placed after all API routes and WebSocket endpoints.

# --- Pydantic Models for Migrated Endpoints ---
class RAGConfigRequest(BaseModel):
    provider_type: str

# --- Migrated HTTP Endpoints ---
@app.post("/rag-config")
async def handle_rag_config_fastapi(request: Request, config_request: RAGConfigRequest):
    """Handles POST request to set the RAG provider type."""
    try:
        provider_type = config_request.provider_type.lower()
        if provider_type not in ("none", "in_memory", "llama_index"):
            logger.warning(f"Invalid provider_type received: {provider_type}")
            raise HTTPException(status_code=400, detail="Invalid provider_type specified. Must be 'none', 'in_memory', or 'llama_index'.")

        app.state.rag_provider_type = provider_type
        logger.info(f"RAG provider type set to: {provider_type}")
        
        # For now, user_id is not handled here as per simplification instructions
        # We can add Optional[UserInDB] = Depends(get_current_user_or_none) later if needed
        user_id = None # Placeholder

        if provider_type == "none":
            update_rag_provider(app, None)

        return JSONResponse({
            "status": "ok",
            "message": f"RAG provider type set to {provider_type}",
            "user_authenticated": user_id is not None
        })
    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.exception(f"Error handling RAG config: {e}")
        raise HTTPException(status_code=500, detail="Error processing RAG configuration.")

@app.post("/upload")
async def handle_upload_fastapi(
    request: Request,
    files: List[UploadFile] = File(...),
    current_user: UserInDB = Depends(get_current_user)
):
    """Handles file uploads to build and activate a RAG provider."""
    logger.info(f"--- Upload Request Start ---")
    logger.info(f"Request Headers: {dict(request.headers)}") # Convert headers to dict for cleaner logging
    logger.info(f"Authenticated User: {current_user.username if current_user else 'None'}")
    if files:
        try:
            filenames = [f.filename for f in files]
            logger.info(f"Received files parameter with filenames: {filenames}")
        except Exception as e:
            logger.error(f"Error accessing filenames from files parameter: {e}")
    else:
        logger.warning("Received 'files' parameter is empty or None.")

    provider_type = getattr(app.state, "rag_provider_type", "none")
    logger.info(f"--- Checking RAG provider type: {provider_type} ---")

    if provider_type not in ("in_memory", "llama_index"):
        logger.warning(f"Upload attempt received but RAG provider type is '{provider_type}'. Upload not allowed.")
        raise HTTPException(status_code=400, detail=f"File upload not supported for RAG provider type '{provider_type}'. Select 'in_memory' or 'llama_index' first via /rag-config.")

    temp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
    logger.info(f"Created temporary directory for uploads: {temp_dir}")
    try:
        texts: List[str] = []
        titles: List[str] = []
        file_paths: List[Path] = []

        for i, file in enumerate(files):
            safe_filename = f"{i}_{Path(file.filename).name}"
            temp_file_path = temp_dir / safe_filename
            logger.info(f"Processing uploaded file: {file.filename} -> {temp_file_path}")
            file_size = 0
            try:
                content = await file.read()
                with open(temp_file_path, "wb") as f:
                    f.write(content)
                file_size = len(content)
                logger.info(f"Saved {file.filename} ({file_size} bytes) to {temp_file_path}")
                file_paths.append(temp_file_path)

                raw_data = temp_file_path.read_bytes()
                extracted_text = extract_text(file.filename, raw_data)
                if not extracted_text:
                    logger.warning(f"No text extracted from {file.filename}, skipping.")
                    continue
                
                chunks = chunk_text(extracted_text)
                if not chunks:
                    logger.warning(f"No chunks created from {file.filename}, skipping.")
                    continue
                
                logger.info(f"Extracted and chunked {file.filename} into {len(chunks)} chunks.")
                texts.extend(chunks)
                titles.extend([file.filename] * len(chunks))
            except ImportError as e:
                logger.error(f"Missing required library for processing {file.filename}: {e}")
                raise HTTPException(status_code=500, detail=f"Missing required library for processing {file.filename}. Please install all required dependencies.")
            except Exception as e:
                logger.exception(f"Error processing/saving file {file.filename}: {e}")
                # Continue with other files

        if not texts:
            logger.error("No text could be extracted or chunked from uploaded files.")
            raise HTTPException(status_code=400, detail="No processable content found in uploaded files.")

        logger.info(f"Total text chunks to process: {len(texts)}")

        openai_client = getattr(app.state, "openai_client", None)
        embedding_model = getattr(app.state, "openai_embedding_model", None)
        if not openai_client or not embedding_model:
            logger.error("OpenAI client or embedding model not found in app.state.")
            raise HTTPException(status_code=500, detail="Backend configuration error.")

        logger.info(f"Generating embeddings using model: {embedding_model}...")
        try:
            if not isinstance(openai_client, openai.AsyncOpenAI):
                logger.error("OpenAI client is not async, cannot await.")
                raise HTTPException(status_code=500, detail="Internal configuration error (OpenAI client type).")
            
            response = await openai_client.embeddings.create(input=texts, model=embedding_model)
            vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
            logger.info(f"Generated {vectors.shape[0]} embeddings with dimension {vectors.shape[1]}.")
        except Exception as e:
            logger.exception("Error generating embeddings.")
            raise HTTPException(status_code=500, detail="Failed to generate embeddings for uploaded content.")

        metadata_list = [
            {"chunk_id": f"{titles[i]}-{i}", "text": texts[i], "title": titles[i]}
            for i in range(len(texts))
        ]

        new_provider: Optional[BaseRAGProvider] = None
        if provider_type == "in_memory":
            logger.info("Initializing InMemoryRAGProvider...")
            try:
                new_provider = InMemoryRAGProvider(
                    openai_client=openai_client,
                    embedding_model=embedding_model,
                    all_metadata=metadata_list,
                    all_vectors=vectors
                )
                logger.info("InMemoryRAGProvider initialized.")
            except Exception as e:
                logger.exception("Failed to initialize InMemoryRAGProvider")
                raise HTTPException(status_code=500, detail="Failed to initialize In-Memory RAG provider.")
        
        elif provider_type == "llama_index":
            try:
                from .scripts.create_llama_graph_index import create_graph_index # Ensure correct import path
                llama_index_persist_dir = temp_dir / "llama_index_data"
                llama_index_persist_dir.mkdir(exist_ok=True)
                
                # Run create_graph_index in a thread pool
                index = await asyncio.to_thread(
                    create_graph_index,
                    source_dir=str(temp_dir), # Pass the directory containing processed files
                    index_dir=str(llama_index_persist_dir)
                )
            
                retriever = index.as_retriever(similarity_top_k=5)
                new_provider = LlamaIndexGraphRAGProvider(
                    openai_client=openai_client,
                    index_dir=llama_index_persist_dir,
                    embedding_model_name=embedding_model, # Use the model name string
                    llm_model_name=app.state.openai_model, # Use the model name string
                )
                new_provider.index = index # Set the loaded/created index
                new_provider.retriever = retriever # Set the retriever
                logger.info("LlamaIndexGraphRAGProvider initialized and configured with PropertyGraphIndex.")
            except Exception as e:
                logger.exception("Failed to initialize or configure LlamaIndexGraphRAGProvider")
                raise HTTPException(status_code=500, detail=f"Failed to initialize LlamaIndex RAG provider: {e}")


        if new_provider:
            update_rag_provider(app, new_provider)
            logger.info(f"Successfully activated {provider_type} RAG provider with uploaded data.")
            
            user_id = current_user.id
            db_conn_upload = None
            try:
                db_conn_upload = open_db_connection() # Get a new connection for this operation
                for i, file_path_obj in enumerate(file_paths): # Iterate over Path objects
                    original_filename = files[i].filename # Get original filename from UploadFile
                    document_id = f"{provider_type}_{original_filename}_{user_id}_{i}" # More unique ID
                    
                    success = associate_document_with_user(
                        db_conn_upload,
                        user_id,
                        document_id,
                        custom_filename=original_filename # Use original filename
                    )
                    if success:
                        logger.info(f"Document {document_id} (file: {original_filename}) associated with user {user_id}")
            except Exception as e:
                logger.exception(f"Error associating documents with user {user_id}: {e}")
            finally:
                if db_conn_upload:
                    db_conn_upload.close()
            
            return JSONResponse({
                "status": "ok",
                "message": f"{provider_type} RAG provider activated with {len(texts)} chunks from {len(files)} files."
            })
        else:
            logger.error("Failed to create a RAG provider instance after processing uploads.")
            raise HTTPException(status_code=500, detail="Failed to create RAG provider instance.")

    except HTTPException as http_exc:
        logger.error(f"HTTPException during upload: Status={http_exc.status_code}, Detail={http_exc.detail}")
        raise  # Re-raise the captured HTTPException
    except Exception as e_main: # Use a distinct variable name for the main try-except
        logger.exception(f"Unhandled exception during main upload processing: {e_main}")
        # It's important to re-raise or return an appropriate HTTP response
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during file upload processing: {e_main}")
    finally:
        logger.info(f"--- Upload Request End (Attempting Cleanup) ---")
        # temp_dir should be defined if the code reached this point after its creation
        if 'temp_dir' in locals() and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e_cleanup: # Use a different variable name for cleanup exception
                logger.error(f"Failed to clean up temporary directory {temp_dir}: {e_cleanup}")


@app.get("/auth-status")
async def handle_auth_status_fastapi(
    request: Request,
    current_user: Optional[UserInDB] = Depends(get_current_user_or_none)
):
    """Returns the current user's authentication status."""
    if current_user:
        return JSONResponse({
            "authenticated": True,
            "user": {
                "id": current_user.id,
                "username": current_user.username,
                "email": current_user.email,
                "role": current_user.role
            }
        })
    else:
        # Provide a generic auth URL or one derived from configuration if available
        # For now, hardcoding as per original app.py
        # In a real app, this might come from env vars or config
        auth_url_base = str(request.base_url).rstrip('/')
        return JSONResponse({
            "authenticated": False,
            "auth_url": f"{auth_url_base}/auth" # Points to the FastAPI auth router
        })

# --- WebSocket Realtime Endpoint ---
@app.websocket("/realtime")
async def websocket_realtime_endpoint(
    websocket: WebSocket,
    current_user: Optional[UserInDB] = Depends(get_current_user_from_websocket_query_param)
):
    if current_user is None:
        logger.warning("WebSocket connection attempt without valid authentication.")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    logger.info(f"WebSocket connection accepted for user: {current_user.username}")

    rtmt_instance = getattr(app.state, "rtmt", None)
    if not rtmt_instance:
        logger.error("RTMiddleTier instance not found in app.state for WebSocket connection.")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Server configuration error")
        return

    try:
        # Pass the authenticated user to _forward_messages if RTMiddleTier needs it
        # For now, assuming _forward_messages primarily needs the websocket object
        await rtmt_instance._forward_messages(websocket)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user: {current_user.username}")
    except Exception as e:
        logger.exception(f"Error in WebSocket connection for user {current_user.username}: {e}")
        # Ensure the WebSocket is closed if not already by WebSocketDisconnect
        if websocket.client_state != WebSocketDisconnect: # This check might not be right, consult FastAPI docs for state
             try:
                 await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
             except RuntimeError: # Can happen if already closed
                 pass
    finally:
        logger.info(f"WebSocket connection for user {current_user.username} finished.")

# --- Mount static files directories ---
# These need to be after API routes and WebSocket routes to ensure they take precedence.

# Mount the root static directory first to serve files like worklets copied from 'public'
# Ensure this comes AFTER specific routes like '/' for index.html if they should override.
# The explicit @app.get("/") route defined earlier should take precedence over this mount.
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR), name="static-root")
    logger.info(f"Mounted root static directory {STATIC_DIR} at '/'.")
else:
    # This warning was already present, but good to keep
    logger.warning(f"Primary static files directory ({STATIC_DIR}) not found. Cannot serve root static files.")


# Mount the more specific /assets directory next
assets_dir = STATIC_DIR / "assets"
if assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets") # Keep this mount
    logger.info(f"Mounted static assets from {assets_dir} at /assets.")
else:
    logger.warning(f"Assets directory not found: {assets_dir}. Path /assets will not serve static files.")
@app.on_event("shutdown")
async def shutdown_event():
    """Handles application shutdown tasks."""
    logger.info("FastAPI application initiating shutdown...")
    
    # Attempt to close OpenAI client if it exists and has a close method
    openai_client = getattr(app.state, "openai_client", None)
    if openai_client and hasattr(openai_client, "close") and asyncio.iscoroutinefunction(openai_client.close):
        try:
            await openai_client.close()
            logger.info("AsyncOpenAI client closed successfully.")
        except Exception as e:
            logger.error(f"Error closing AsyncOpenAI client: {e}")
    elif openai_client and hasattr(openai_client, "close"): # For synchronous close, though AsyncOpenAI uses async
        try:
            openai_client.close() # This might be sync depending on the actual client
            logger.info("OpenAI client (sync close attempted) closed successfully.")
        except Exception as e:
            logger.error(f"Error closing OpenAI client (sync close attempted): {e}")
    else:
        logger.info("OpenAI client not found or does not have a close method.")
        
    logger.info("FastAPI application shutdown complete.")

# --- Application Entry Point (for uvicorn) ---
if __name__ == "__main__":
    host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("BACKEND_PORT", 8765))

    logger.info(f"Starting FastAPI application server with Uvicorn on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True) # Added reload=True for dev