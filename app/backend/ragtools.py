import re
import logging
import numpy as np
import openai
import asyncio
from typing import Any
from sklearn.metrics.pairwise import cosine_similarity

from rtmt import RTMiddleTier, Tool, ToolResult, ToolResultDirection

logger = logging.getLogger(__name__)

_search_tool_schema = {
    "type": "function",
    "name": "search",
    "description": "Search the knowledge base for information relevant to the user query. Results are formatted as a source name first in square brackets, followed by the text content.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query derived from the user's request"
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

_grounding_tool_schema = {
    "type": "function",
    "name": "report_grounding",
    "description": "Report the specific sources (by their chunk_id) from the knowledge base that were actually used to formulate the answer. Sources appear in square brackets in search results.",
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "The chunk_id of a source used in the answer."
                },
                "description": "List of source chunk_ids used to formulate the answer. Only include sources directly used."
            }
        },
        "required": ["sources"],
        "additionalProperties": False
    }
}

async def _search_tool(
    openai_client: openai.OpenAI,
    embedding_model: str,
    all_metadata: list[dict],
    all_vectors: np.ndarray,
    args: Any) -> ToolResult:

    query = args.get('query')
    if not query:
        logger.error("Search tool called without a query.")
        return ToolResult("Error: Query parameter is missing.", ToolResultDirection.TO_SERVER)

    logger.info(f"Searching in-memory for '{query}'...")

    if all_vectors is None or len(all_metadata) == 0 or all_vectors.size == 0:
         logger.warning("Search tool called but in-memory RAG index is not loaded or empty.")
         return ToolResult("Knowledge base is not available.", ToolResultDirection.TO_SERVER)

    try:
        # 1. Get query embedding using asyncio.to_thread for the sync SDK call
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, # Use default executor
            lambda: openai_client.embeddings.create(
                input=[query],
                model=embedding_model
            )
        )
        query_vector = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
        # Note: OpenAI embeddings (v2+) are pre-normalized to length 1

        # 2. Calculate cosine similarities
        # Ensure vectors are compatible shapes
        if query_vector.shape[1] != all_vectors.shape[1]:
             logger.error(f"Query vector dimension ({query_vector.shape[1]}) does not match index dimension ({all_vectors.shape[1]}).")
             return ToolResult("Error: Embedding dimension mismatch.", ToolResultDirection.TO_SERVER)

        similarities = cosine_similarity(query_vector, all_vectors)[0]

        # 3. Get top K results
        K = 5 # Number of results to return
        # Ensure K is not larger than the number of available vectors
        actual_k = min(K, len(similarities))
        if actual_k == 0:
             return ToolResult("No results found in the knowledge base.", ToolResultDirection.TO_SERVER)

        top_k_indices = np.argsort(similarities)[-actual_k:][::-1]

        # 4. Format results
        result_text = ""
        for i in top_k_indices:
            metadata = all_metadata[i]
            score = similarities[i]
            # Use chunk_id as the source identifier
            result_text += f"[{metadata['chunk_id']}]: {metadata['text']}\n-----\n"
            logger.info(f"  Found: {metadata['chunk_id']} (Score: {score:.4f})")

        if not result_text:
             result_text = "No relevant information found in the knowledge base."

        return ToolResult(result_text, ToolResultDirection.TO_SERVER)

    except openai.APIError as e:
        logger.error(f"OpenAI API error during embedding query: {e}")
        return ToolResult(f"Error reaching embedding service: {e}", ToolResultDirection.TO_SERVER)
    except Exception as e:
        logger.exception(f"Error during in-memory search: {e}")
        return ToolResult(f"An internal error occurred during search.", ToolResultDirection.TO_SERVER)

async def _report_grounding_tool(all_metadata: list[dict], args: Any) -> ToolResult:
    sources_requested = args.get("sources", [])
    logger.info(f"Grounding sources requested: {sources_requested}")

    if not all_metadata:
        logger.warning("Grounding tool called but no metadata loaded.")
        return ToolResult({"sources": []}, ToolResultDirection.TO_CLIENT)

    docs = []
    # Create a quick lookup map for efficiency for potentially repeated calls
    # Build it once if used frequently or just scan if metadata list is small
    metadata_map = {item['chunk_id']: item for item in all_metadata}

    found_ids = set()
    for source_id in sources_requested:
        if source_id in metadata_map and source_id not in found_ids:
            item = metadata_map[source_id]
            docs.append({"chunk_id": item['chunk_id'], "title": item['title'], "chunk": item['text']})
            found_ids.add(source_id)
        else:
             if source_id in found_ids:
                 logger.debug(f"  Duplicate grounding source ID requested: {source_id}")
             else:
                 logger.warning(f"  Grounding source ID not found in metadata: {source_id}")

    logger.info(f"Grounding result contains {len(docs)} documents.")
    # Send structured data to client
    return ToolResult({"sources": docs}, ToolResultDirection.TO_CLIENT)

def attach_rag_tools(rtmt: RTMiddleTier,
    openai_client: openai.OpenAI,
    embedding_model: str,
    all_metadata: list[dict],
    all_vectors: np.ndarray
    ) -> None:

    # No Azure search client needed

    # Check if data is actually present before attaching
    if not all_metadata or all_vectors is None or all_vectors.size == 0:
        logger.warning("Attempted to attach RAG tools, but index data is missing or empty. Skipping.")
        return

    # Update lambda to pass new arguments correctly
    rtmt.tools["search"] = Tool(
        schema=_search_tool_schema,
        target=lambda args: _search_tool(
            openai_client, embedding_model, all_metadata, all_vectors, args
        )
    )
    rtmt.tools["report_grounding"] = Tool(
         schema=_grounding_tool_schema,
         target=lambda args: _report_grounding_tool(all_metadata, args)
    )
    logger.info(f"Attached in-memory RAG tools ({len(all_metadata)} items).")
