import os
import json
import numpy as np
import openai
import asyncio # Required for potential async operations if needed later, though embedding is sync
from dotenv import load_dotenv
import logging
import time
from pypdf import PdfReader # Added

# --- Configuration ---
# Assuming this script is in app/backend and data is in app/data
# Adjust paths if your structure is different
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(APP_DIR, ".env")
# Use relative path for DATA_DIR for better portability
DATA_DIR = os.path.join(APP_DIR, "..", "..", "data")
OUTPUT_METADATA_FILE = os.path.join(APP_DIR, "rag_data.jsonl")
OUTPUT_VECTOR_FILE = os.path.join(APP_DIR, "rag_vectors.npy")

# Load environment variables (especially OPENAI_API_KEY)
print(f"Loading environment variables from: {ENV_PATH}")
load_dotenv(dotenv_path=ENV_PATH)

# Embedding model config - make sure this matches your intended model
# Update OPENAI_EMBEDDING_MODEL in .env or set default here
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
# Get dimension for your chosen model (e.g., 1536 for text-embedding-3-small, 1536 for ada-002, 3072 for text-embedding-3-large)
# You might need to manually set this based on the model you choose.
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    "text-embedding-3-large": 3072,
}
EMBEDDING_DIM = MODEL_DIMENSIONS.get(EMBEDDING_MODEL, 1536) # Default fallback

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
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Make sure it's set in app/backend/.env")

# It's good practice to set timeout for production code
client = openai.OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT)

# --- Simple Text Splitter (replace with langchain if preferred) ---
def split_text(text, chunk_size, chunk_overlap):
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        # Ensure we don't create empty overlaps if overlap is large
        if start < 0: start = 0
    return chunks

# --- Helper to get embeddings with retries ---
def get_embeddings_with_retry(texts, model, max_retries=3, initial_delay=1):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
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
            raise # Re-raise other unexpected errors
    logger.error("Failed to get embeddings after multiple retries.")
    return None


# --- Main Processing Logic ---
def create_index():
    all_metadata = []
    all_vectors = []
    doc_texts_to_embed = []
    doc_metadata_to_add = []

    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory not found: {DATA_DIR}")
        return

    logger.info(f"Processing documents in {DATA_DIR} using model '{EMBEDDING_MODEL}'...")
    filenames = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f)) and not f.startswith('.')]
    logger.info(f"Found {len(filenames)} files to process.")

    total_chunks = 0
    processed_files = 0

    for filename in filenames:
        filepath = os.path.join(DATA_DIR, filename)
        logger.info(f"  Processing {filename}...")
        content = ""
        try:
            # --- Modified File Reading --- 
            if filename.lower().endswith('.pdf'):
                try:
                    reader = PdfReader(filepath)
                    text_parts = []
                    for page in reader.pages:
                        text_parts.append(page.extract_text() or "") # Add empty string if page has no text
                    content = "\n".join(text_parts)
                    if not content.strip():
                         logger.warning(f"    Extracted no text content from PDF: {filename}")
                except Exception as pdf_error:
                    logger.error(f"    Error reading PDF file {filename}: {pdf_error}")
                    continue # Skip to next file on PDF read error
            elif filename.lower().endswith(('.txt', '.md', '.jsonl', '.json', '.csv')): # Add other text formats if needed
                 with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                 logger.warning(f"    Skipping unsupported file type: {filename}")
                 continue # Skip to next file
            # --- End Modified File Reading ---

            if not content:
                logger.warning(f"    No content read from file: {filename}")
                continue # Skip if no content was extracted

            # Split text into chunks
            chunks = split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            logger.info(f"    Split into {len(chunks)} chunks.")
            total_chunks += len(chunks)

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                chunk_id = f"{filename}-{i}"
                doc_texts_to_embed.append(chunk)
                doc_metadata_to_add.append({"chunk_id": chunk_id, "text": chunk, "title": filename})

                # Process in batches
                if len(doc_texts_to_embed) >= BATCH_SIZE:
                    logger.info(f"    Generating embeddings for batch of {len(doc_texts_to_embed)}...")
                    embeddings = get_embeddings_with_retry(doc_texts_to_embed, model=EMBEDDING_MODEL)
                    if embeddings:
                        all_vectors.extend(embeddings)
                        all_metadata.extend(doc_metadata_to_add)
                    else:
                        logger.error(f"    Failed to get embeddings for batch starting with chunk from {filename}. Skipping batch.")
                    # Clear batch regardless of success/failure to proceed
                    doc_texts_to_embed = []
                    doc_metadata_to_add = []

            processed_files += 1

        except Exception as e:
            # Catch errors during splitting or batch preparation
            logger.error(f"    Error processing content of file {filename}: {e}")

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
    logger.info(f"Generated {len(all_vectors)} vectors for {len(all_metadata)} non-empty chunks (expected total: {total_chunks}).")

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
        return

    logger.info(f"Saving vectors to {OUTPUT_VECTOR_FILE}...")
    try:
        vectors_np = np.array(all_vectors, dtype=np.float32)
        # Dimension check before saving
        if vectors_np.ndim != 2 or vectors_np.shape[1] != EMBEDDING_DIM:
             logger.error(f"Vector array shape is incorrect ({vectors_np.shape}). Expected (N, {EMBEDDING_DIM}). Aborting save.")
             # Clean up metadata file if vector save fails
             if os.path.exists(OUTPUT_METADATA_FILE): os.remove(OUTPUT_METADATA_FILE)
             return

        np.save(OUTPUT_VECTOR_FILE, vectors_np)
    except Exception as e:
        logger.error(f"Error converting vectors to NumPy array or saving: {e}")
        # Clean up metadata file if vector save fails
        if os.path.exists(OUTPUT_METADATA_FILE): os.remove(OUTPUT_METADATA_FILE)
        return

    logger.info("Indexing complete. Files saved successfully.")

if __name__ == "__main__":
    create_index() 