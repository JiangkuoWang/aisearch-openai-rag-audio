import logging
import os
from pathlib import Path
import json
import numpy as np
import openai
import sys
from typing import Optional, List, Dict, Any
import asyncio
import tempfile
import shutil # Added for directory cleanup new_pro

from aiohttp import web
from dotenv import load_dotenv

# Local imports
from rtmt import RTMiddleTier
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

        # If 'none', detach tools immediately.
        if provider_type == "none":
            update_rag_provider(app, None)
        # For other types, we wait for an upload to activate the provider.
        # If a provider of the *same type* was already active, it remains until replaced by upload.
        # If the type changes, we could optionally clear the old provider here:
        # else:
        #     if app.get("rag_provider") and type(app.get("rag_provider")).__name__.lower().replace("ragprovider","") != provider_type.replace("_",""):
        #          update_rag_provider(app, None) # Clear if type differs

        return web.json_response({"status": "ok", "message": f"RAG provider type set to {provider_type}"})
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
            # 暂不支持动态构建 LlamaIndex，退回使用 InMemoryRAGProvider
            logger.warning("LlamaIndex 模式上传暂不支持，使用 InMemoryRAGProvider 替代")
            try:
                new_provider = InMemoryRAGProvider(
                    openai_client=openai_client,
                    embedding_model=embedding_model,
                    all_metadata=metadata_list,
                    all_vectors=vectors
                )
                logger.info("Fallback to InMemoryRAGProvider for llama_index mode.")
            except Exception as e:
                logger.exception("Failed to initialize fallback InMemoryRAGProvider for llama_index mode.")
                return web.HTTPInternalServerError(text="Failed to initialize fallback RAG provider.")

        # Activate the new provider
        if new_provider:
            update_rag_provider(app, new_provider)
            logger.info(f"Successfully activated {provider_type} RAG provider with uploaded data.")
            # Clean up temp dir *only* after successful provider activation
            # For LlamaIndex, decide if you need to keep the persisted index dir
            # If LlamaIndexGraphProvider loads from graph_index_dir on init,
            # and initialize builds it if not present, we might keep llama_index_persist_dir
            # but remove the original uploaded files.
            # Simple cleanup for now: remove the whole temp dir. Adjust if needed.
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


# --- Main Application Setup ---
async def create_app():
    app = web.Application()
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

    # Add HTTP routes for RAG config and upload
    app.router.add_post("/rag-config", handle_rag_config)
    app.router.add_post("/upload", handle_upload)
    logger.info("Added HTTP routes: /rag-config (POST), /upload (POST)")


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
