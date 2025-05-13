import os
import json
import numpy as np
import openai
import asyncio # Required for potential async operations if needed later, though embedding is sync
# from dotenv import load_dotenv # Removed
import logging
import time
from pypdf import PdfReader # Added
import sys # Added to adjust path for sibling imports if needed
from app.backend.config import config_service # Added
from pathlib import Path # Added

# --- Configuration ---
# Determine the script's directory and the backend directory
SCRIPT_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = SCRIPT_DIR.parent # Assumes script is in app/backend/scripts

# Path to the .env file in the backend directory - Removed
# ENV_PATH = os.path.join(BACKEND_DIR, ".env") # Removed

# Load environment variables BEFORE accessing them - Removed
# print(f"Loading environment variables from: {ENV_PATH}") # Removed
# if not os.path.exists(ENV_PATH): # Removed
#     print(f"Warning: .env file not found at {ENV_PATH}. Relying on existing environment variables.") # Removed
# load_dotenv(dotenv_path=ENV_PATH) # Removed

# Get paths from config_service, resolved relative to BACKEND_DIR
DATA_SOURCE_DIR = (BACKEND_DIR / config_service.settings.DATA_SOURCE_DIR_RELATIVE).resolve()
# RAG_DATA_DIR is not directly used for output files, but individual file paths are constructed
# RAG_DATA_DIR = (BACKEND_DIR / config_service.settings.RAG_DATA_DIR_RELATIVE).resolve() # Not strictly needed if specific file paths are used

OUTPUT_METADATA_FILE = (BACKEND_DIR / config_service.settings.RAG_METADATA_FILE_RELATIVE).resolve()
OUTPUT_VECTOR_FILE = (BACKEND_DIR / config_service.settings.RAG_VECTOR_FILE_RELATIVE).resolve()

# Ensure the output directory exists
Path(OUTPUT_METADATA_FILE).parent.mkdir(parents=True, exist_ok=True)
Path(OUTPUT_VECTOR_FILE).parent.mkdir(parents=True, exist_ok=True) # Also ensure vector file's parent dir

# Embedding model config
EMBEDDING_MODEL = config_service.settings.OPENAI_EMBEDDING_MODEL

# Get dimension for the chosen model
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    "text-embedding-3-large": 3072,
}
EMBEDDING_DIM = MODEL_DIMENSIONS.get(EMBEDDING_MODEL)
if EMBEDDING_DIM is None:
     # Fallback or error if model not in dict
     print(f"Warning: Unknown embedding model '{EMBEDDING_MODEL}' from config_service.settings. Using default dimension 1536. Add model to MODEL_DIMENSIONS if needed.")
     EMBEDDING_DIM = 1536


# Text splitting config
CHUNK_SIZE = 1000 # Max characters per chunk
CHUNK_OVERLAP = 200 # Characters overlap between chunks

# Batching config for OpenAI API
BATCH_SIZE = 16 # Number of texts to embed in one API call
REQUEST_TIMEOUT = 30 # Timeout for API requests in seconds

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialize OpenAI Client ---
api_key_secret = config_service.settings.OPENAI_API_KEY
if not api_key_secret:
    raise ValueError("OPENAI_API_KEY not found in config_service.settings.")
api_key = api_key_secret.get_secret_value() if api_key_secret else None

client = openai.OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT)

# --- Simple Text Splitter ---
def split_text(text, chunk_size, chunk_overlap):
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start < 0: start = 0 # Ensure start index isn't negative
    return chunks

# --- Helper to get embeddings with retries ---
def get_embeddings_with_retry(texts, model, max_retries=3, initial_delay=1):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=texts, model=model)
            # Check if the response format is as expected
            if response and response.data:
                return [item.embedding for item in response.data]
            else:
                logger.error(f"Unexpected response format from OpenAI embeddings: {response}")
                return None # Indicate failure
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded, retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            delay *= 2 # Exponential backoff
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            logger.error(f"An unexpected error occurred during embedding: {e}")
            # Optionally re-raise specific errors if needed, otherwise log and fail
            break # Exit retry loop on unexpected error
    logger.error("Failed to get embeddings after multiple retries.")
    return None


# --- Main Processing Logic ---
def create_index():
    all_metadata = []
    all_vectors = []
    doc_texts_to_embed = []
    doc_metadata_to_add = []

    if not os.path.exists(DATA_SOURCE_DIR):
        logger.error(f"Data source directory not found: {DATA_SOURCE_DIR}")
        return

    logger.info(f"Using Data Source Directory: {DATA_SOURCE_DIR}")
    logger.info(f"Output Metadata File: {OUTPUT_METADATA_FILE}")
    logger.info(f"Output Vector File: {OUTPUT_VECTOR_FILE}")
    logger.info(f"Using Embedding Model: '{EMBEDDING_MODEL}' (Dimensions: {EMBEDDING_DIM})")

    try:
        filenames = [f for f in os.listdir(DATA_SOURCE_DIR) if os.path.isfile(os.path.join(DATA_SOURCE_DIR, f)) and not f.startswith('.')]
    except OSError as e:
        logger.error(f"Error listing files in data source directory {DATA_SOURCE_DIR}: {e}")
        return

    logger.info(f"Found {len(filenames)} files to process.")

    total_chunks = 0
    processed_files = 0

    for filename in filenames:
        filepath = os.path.join(DATA_SOURCE_DIR, filename)
        logger.info(f"  Processing {filename}...")
        content = ""
        try:
            # --- Modified File Reading ---
            if filename.lower().endswith('.pdf'):
                try:
                    reader = PdfReader(filepath)
                    text_parts = [page.extract_text() or "" for page in reader.pages]
                    content = "\n".join(text_parts)
                    if not content.strip():
                         logger.warning(f"    Extracted no text content from PDF: {filename}")
                except Exception as pdf_error:
                    logger.error(f"    Error reading PDF file {filename}: {pdf_error}")
                    continue # Skip to next file
            elif filename.lower().endswith(('.txt', '.md', '.jsonl', '.json', '.csv')):
                 try:
                     with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                 except Exception as read_error:
                     logger.error(f"    Error reading text file {filename}: {read_error}")
                     continue # Skip to next file
            else:
                 logger.warning(f"    Skipping unsupported file type: {filename}")
                 continue # Skip to next file
            # --- End Modified File Reading ---

            if not content or not content.strip():
                logger.warning(f"    No content read or empty content for file: {filename}")
                continue # Skip if no content was extracted

            # Split text into chunks
            try:
                chunks = split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            except ValueError as split_error:
                 logger.error(f"    Error splitting text for file {filename}: {split_error}")
                 continue # Skip file if splitting fails

            logger.info(f"    Split into {len(chunks)} chunks.")
            chunk_count_for_file = 0

            for i, chunk in enumerate(chunks):
                cleaned_chunk = chunk.strip()
                if not cleaned_chunk:
                    continue # Skip empty chunks after stripping

                chunk_id = f"{filename}-{i}" # Use original filename and index for ID
                doc_texts_to_embed.append(cleaned_chunk)
                # Store only necessary metadata for JSONL
                doc_metadata_to_add.append({"chunk_id": chunk_id, "text": cleaned_chunk, "title": filename})
                chunk_count_for_file += 1

                # Process in batches
                if len(doc_texts_to_embed) >= BATCH_SIZE:
                    logger.info(f"    Generating embeddings for batch of {len(doc_texts_to_embed)}...")
                    embeddings = get_embeddings_with_retry(doc_texts_to_embed, model=EMBEDDING_MODEL)
                    if embeddings:
                        all_vectors.extend(embeddings)
                        all_metadata.extend(doc_metadata_to_add)
                    else:
                        logger.error(f"    Failed to get embeddings for batch starting with chunk from {filename}. Skipping batch.")
                        # Consider how to handle this failure - maybe stop processing? For now, just skip.
                    # Clear batch regardless of success/failure
                    doc_texts_to_embed = []
                    doc_metadata_to_add = []

            total_chunks += chunk_count_for_file # Add count of non-empty chunks for the file
            processed_files += 1

        except Exception as e:
            # Catch broader errors during file processing loop
            logger.exception(f"    Unexpected error processing file {filename}: {e}")
            # Reset batch for safety before next file
            doc_texts_to_embed = []
            doc_metadata_to_add = []

    # Process any remaining documents
    if doc_texts_to_embed:
        logger.info(f"    Generating embeddings for final batch of {len(doc_texts_to_embed)}...")
        embeddings = get_embeddings_with_retry(doc_texts_to_embed, model=EMBEDDING_MODEL)
        if embeddings:
            all_vectors.extend(embeddings)
            all_metadata.extend(doc_metadata_to_add)
        else:
            logger.error("    Failed to get embeddings for final batch. These chunks will be missing.")

    logger.info(f"Finished processing {processed_files} files.")
    logger.info(f"Generated {len(all_vectors)} vectors for {len(all_metadata)} non-empty chunks (total non-empty chunks processed: {total_chunks}).")

    if not all_vectors or not all_metadata:
        logger.error("No vectors or metadata were generated. Aborting save.")
        return

    # --- Save to Files ---
    logger.info(f"Saving metadata to {OUTPUT_METADATA_FILE}...")
    try:
        with open(OUTPUT_METADATA_FILE, 'w', encoding='utf-8') as f_meta:
            for item in all_metadata:
                f_meta.write(json.dumps(item) + '\n')
    except IOError as e:
        logger.error(f"Error writing metadata file: {e}")
        return # Don't proceed to save vectors if metadata failed

    logger.info(f"Saving vectors to {OUTPUT_VECTOR_FILE}...")
    try:
        vectors_np = np.array(all_vectors, dtype=np.float32)
        # Dimension check before saving
        if vectors_np.ndim != 2 or vectors_np.shape[1] != EMBEDDING_DIM:
             logger.error(f"Vector array shape is incorrect ({vectors_np.shape}). Expected (N, {EMBEDDING_DIM}). Aborting save.")
             # Attempt to clean up the possibly incomplete metadata file
             try:
                 if os.path.exists(OUTPUT_METADATA_FILE): os.remove(OUTPUT_METADATA_FILE)
                 logger.info(f"Cleaned up metadata file due to vector save error: {OUTPUT_METADATA_FILE}")
             except OSError as remove_err:
                 logger.error(f"Error cleaning up metadata file {OUTPUT_METADATA_FILE}: {remove_err}")
             return

        np.save(OUTPUT_VECTOR_FILE, vectors_np)
    except Exception as e:
        logger.error(f"Error converting vectors to NumPy array or saving: {e}")
        # Attempt to clean up the metadata file
        try:
            if os.path.exists(OUTPUT_METADATA_FILE): os.remove(OUTPUT_METADATA_FILE)
            logger.info(f"Cleaned up metadata file due to vector save error: {OUTPUT_METADATA_FILE}")
        except OSError as remove_err:
            logger.error(f"Error cleaning up metadata file {OUTPUT_METADATA_FILE}: {remove_err}")
        return

    logger.info("Indexing complete. Files saved successfully.")

if __name__ == "__main__":
    create_index() 