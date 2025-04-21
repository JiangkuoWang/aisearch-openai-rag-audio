import logging
import os
from pathlib import Path
import json
import numpy as np
import openai

from aiohttp import web
from dotenv import load_dotenv

from ragtools import attach_rag_tools
from rtmt import RTMiddleTier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicerag")

# Define RAG file paths (assuming they are in the same directory as app.py)
RAG_METADATA_FILE = "rag_data.jsonl"
RAG_VECTOR_FILE = "rag_vectors.npy"

async def create_app():
    if not os.environ.get("RUNNING_IN_PRODUCTION"):
        logger.info("Running in development mode, loading from .env file")
        load_dotenv()

    # --- Load OpenAI Key and Model for Realtime API ---
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
    openai_embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small") # Added for RAG

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # --- Load RAG Data into Memory ---
    rag_metadata = []
    rag_vectors = None
    if os.path.exists(RAG_METADATA_FILE) and os.path.exists(RAG_VECTOR_FILE):
        logger.info(f"Loading RAG data from {RAG_METADATA_FILE} and {RAG_VECTOR_FILE}...")
        try:
            with open(RAG_METADATA_FILE, 'r', encoding='utf-8') as f:
                rag_metadata = [json.loads(line) for line in f]
            rag_vectors = np.load(RAG_VECTOR_FILE)
            if len(rag_metadata) != rag_vectors.shape[0]:
                logger.error("Mismatch between number of metadata entries and vectors. RAG will be disabled.")
                rag_metadata = []
                rag_vectors = None
            else:
                logger.info(f"Loaded {len(rag_metadata)} chunks for RAG.")
                # Optional: Normalize vectors here if needed and not done during indexing
        except Exception as e:
            logger.exception(f"Error loading RAG data: {e}. RAG will be disabled.")
            rag_metadata = []
            rag_vectors = None
    else:
        logger.warning("RAG data files not found. RAG will be disabled.")

    # --- Initialize OpenAI Client for RAG (can potentially share with Realtime if needed) ---
    # Note: The RTMiddleTier uses the API key directly for WebSocket auth,
    # while RAG tools need an OpenAI client object for the embeddings API.
    openai_client_for_rag = openai.OpenAI(api_key=openai_api_key)

    app = web.Application()

    # --- Initialize RTMiddleTier (as modified previously) ---
    rtmt = RTMiddleTier(
        openai_api_key=openai_api_key,
        model=openai_model,
        voice_choice=os.environ.get("AZURE_OPENAI_REALTIME_VOICE_CHOICE") or "alloy"
    )
    
    # 先禁用RAG功能，先将API换成openai官方的api。所以将原有的提示词也进行更改。
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

    # --- Attach RAG tools if data loaded successfully ---
    if rag_metadata and rag_vectors is not None:
        logger.info("Attaching in-memory RAG tools...")
        attach_rag_tools(
            rtmt=rtmt,
            openai_client=openai_client_for_rag, # Pass the client
            embedding_model=openai_embedding_model, # Pass model name
            all_metadata=rag_metadata, # Pass loaded data
            all_vectors=rag_vectors # Pass loaded data
        )
        # Ensure RTMiddleTier knows tools are available
        # This might involve setting tool_choice logic within RTMiddleTier or
        # ensuring its _process_message_to_server enables 'auto' if rtmt.tools is populated.
    else:
        logger.info("Skipping RAG tool attachment as data was not loaded.")

    rtmt.attach_to_app(app, "/realtime")

    current_directory = Path(__file__).parent
    app.add_routes([web.get('/', lambda _: web.FileResponse(current_directory / 'static/index.html'))])
    app.router.add_static('/', path=current_directory / 'static', name='static')
    
    return app

if __name__ == "__main__":
    host = "localhost"
    port = 8765
    web.run_app(create_app(), host=host, port=port)
