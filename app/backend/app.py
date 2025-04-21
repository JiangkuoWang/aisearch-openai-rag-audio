import logging
import os
from pathlib import Path
import json
import numpy as np
import openai
import sys # Add sys for path manipulation if needed
from typing import Optional

from aiohttp import web
from dotenv import load_dotenv

# Local imports
from ragtools import attach_rag_tools
from rtmt import RTMiddleTier
from rag_providers.base import BaseRAGProvider # Import base provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicerag")

# Define backend directory for constructing absolute paths from .env
BACKEND_DIR = Path(__file__).parent.resolve()

async def create_app():
    if not os.environ.get("RUNNING_IN_PRODUCTION"): # Check env var first
        logger.info("Running in development mode, loading from .env file")
        env_path = BACKEND_DIR / ".env"
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path)
            logger.info(f"Loaded environment variables from: {env_path}")
        else:
            logger.warning(f".env file not found at {env_path}, relying on existing environment variables.")

    # --- Load OpenAI Key and Model --- 
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
    openai_embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # --- RAG Configuration --- 
    rag_provider_type = os.environ.get("RAG_PROVIDER_TYPE", "none").lower()
    # Get absolute paths for RAG data files relative to backend dir
    rag_metadata_file_path = BACKEND_DIR / os.environ.get("RAG_METADATA_FILE", "rag_data/rag_data.jsonl")
    rag_vector_file_path = BACKEND_DIR / os.environ.get("RAG_VECTOR_FILE", "rag_data/rag_vectors.npy")

    rag_provider: Optional[BaseRAGProvider] = None
    rag_metadata = []
    rag_vectors = None

    # --- Initialize OpenAI Client (used by both RTMiddleTier and RAG Provider) ---
    # Note: RTMiddleTier uses the key directly, InMemory provider needs client object
    openai_client = openai.OpenAI(api_key=openai_api_key)

    # --- Load RAG Data and Instantiate Provider --- 
    if rag_provider_type == "in_memory":
        logger.info(f"Attempting to load In-Memory RAG data...")
        logger.info(f"  Metadata file: {rag_metadata_file_path}")
        logger.info(f"  Vector file: {rag_vector_file_path}")
        if rag_metadata_file_path.exists() and rag_vector_file_path.exists():
            try:
                with open(rag_metadata_file_path, 'r', encoding='utf-8') as f:
                    rag_metadata = [json.loads(line) for line in f]
                rag_vectors = np.load(rag_vector_file_path)

                if len(rag_metadata) != rag_vectors.shape[0]:
                    logger.error("Mismatch between number of metadata entries and vectors. In-Memory RAG disabled.")
                    rag_metadata = []
                    rag_vectors = None
                else:
                    logger.info(f"Loaded {len(rag_metadata)} chunks for In-Memory RAG.")
                    # Instantiate the provider *only if data loaded successfully*
                    try:
                        # Dynamically import the provider class
                        from rag_providers.in_memory import InMemoryRAGProvider
                        rag_provider = InMemoryRAGProvider(
                            openai_client=openai_client,
                            embedding_model=openai_embedding_model,
                            all_metadata=rag_metadata,
                            all_vectors=rag_vectors
                        )
                        # Optional: Call async initialize if the provider needs it
                        # await rag_provider.initialize()
                        logger.info("In-Memory RAG Provider Initialized.")
                    except ImportError as e:
                         logger.error(f"Failed to import InMemoryRAGProvider: {e}. RAG disabled.")
                    except Exception as e:
                        logger.exception(f"Error initializing InMemoryRAGProvider: {e}. RAG disabled.")
                        rag_provider = None # Ensure provider is None on error

            except Exception as e:
                logger.exception(f"Error loading RAG data files: {e}. In-Memory RAG disabled.")
                rag_metadata = []
                rag_vectors = None
        else:
            logger.warning("One or both RAG data files not found. In-Memory RAG disabled.")
    elif rag_provider_type == "azure_search":
         # Placeholder for Azure Search RAG Provider initialization
         logger.warning("Azure Search RAG provider selected but not yet implemented in app.py. RAG disabled.")
         # Example:
         # try:
         #     from rag_providers.azure_search import AzureSearchRAGProvider
         #     search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
         #     search_index = os.environ.get("AZURE_SEARCH_INDEX")
         #     search_key = os.environ.get("AZURE_SEARCH_API_KEY") # Use appropriate credential method
         #     if not all([search_endpoint, search_index, search_key]):
         #         raise ValueError("Missing Azure Search environment variables for RAG.")
         #     rag_provider = AzureSearchRAGProvider(endpoint=search_endpoint, index_name=search_index, credential=search_key, embedding_model=openai_embedding_model, openai_client=openai_client)
         #     await rag_provider.initialize()
         #     logger.info("Azure Search RAG Provider Initialized.")
         # except Exception as e:
         #      logger.exception(f"Error initializing AzureSearchRAGProvider: {e}. RAG disabled.")
         #      rag_provider = None
    elif rag_provider_type != "none":
        logger.warning(f"Unsupported RAG_PROVIDER_TYPE: '{rag_provider_type}'. RAG disabled.")
    else:
        logger.info("RAG_PROVIDER_TYPE is 'none' or not set. RAG is disabled.")


    # --- Initialize RTMiddleTier --- 
    rtmt = RTMiddleTier(
        openai_api_key=openai_api_key,
        model=openai_model,
        voice_choice=os.environ.get("AZURE_OPENAI_REALTIME_VOICE_CHOICE") or "alloy"
    )

    # Configure System Prompt (remains the same)
    rtmt.system_message = """
        You are a helpful assistant. Only answer questions based on information you searched in the knowledge base, accessible with the 'search' tool.
        The user is listening to answers with audio, so it's *super* important that answers are as short as possible, a single sentence if at all possible.
        Never read file names or source names or keys out loud.
        Always use the following step-by-step instructions to respond:
        1. Always use the 'search' tool to check the knowledge base before answering a question.
        2. Always use the 'report_grounding' tool to report the source of information from the knowledge base.
        3. Produce an answer that's as short as possible. If the answer isn't in the knowledge base, say you don't know.
    """.strip()
    # rtmt.system_message = "You are a helpful assistant."

    # --- Attach RAG tools *if* a provider was successfully initialized ---
    if rag_provider:
        logger.info(f"Attaching RAG tools using {type(rag_provider).__name__}...")
        attach_rag_tools(
            rtmt=rtmt,
            rag_provider=rag_provider # Pass the provider instance
        )
    else:
        logger.info("Skipping RAG tool attachment as no RAG provider was initialized.")

    # --- Attach WebSocket and Static Routes ---
    rtmt.attach_to_app(app, "/realtime")

    static_dir = BACKEND_DIR / 'static' # Use Path object
    if not static_dir.exists():
        logger.warning(f"Static directory not found at {static_dir}. Frontend may not load.")
    else:
        logger.info(f"Serving static files from: {static_dir}")
        app.add_routes([web.get('/', lambda _: web.FileResponse(static_dir / 'index.html'))])
        app.router.add_static('/', path=static_dir, name='static')

    return app

if __name__ == "__main__":
    app = web.Application() # Create app instance before calling create_app
    async def main():
        return await create_app()

    host = os.environ.get("BACKEND_HOST", "localhost")
    port = int(os.environ.get("BACKEND_PORT", 8765))
    logger.info(f"Starting application server on {host}:{port}")
    web.run_app(main(), host=host, port=port)
