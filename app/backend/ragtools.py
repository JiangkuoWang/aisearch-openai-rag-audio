import re
import logging
import numpy as np
import openai
import asyncio
from typing import Any, List, Dict

from .rtmt import RTMiddleTier, Tool, ToolResult, ToolResultDirection
from .rag_providers.base import BaseRAGProvider

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

async def _execute_search(rag_provider: BaseRAGProvider, args: Any) -> ToolResult:
    """Handles the 'search' tool call by delegating to the RAG provider."""
    query = args.get('query')
    if not query:
        logger.error("Search tool called without a query argument.")
        return ToolResult("Error: Query parameter is missing.", ToolResultDirection.TO_SERVER)

    try:
        # Delegate search to the provider
        search_results: List[Dict[str, Any]] = await rag_provider.search(query=query, top_k=5) # Use top_k=5 as before

        # Format results for the LLM
        if not search_results:
            result_text = "No relevant information found in the knowledge base."
            logger.info(f"Search for '{query}' returned no results.")
        else:
            result_text = ""
            for result in search_results:
                chunk_id = result.get("chunk_id", "unknown_id")
                text = result.get("text", "")
                # Ensure the format matches what the LLM expects based on the prompt
                result_text += f"[{chunk_id}]: {text}\n-----\n"
            logger.info(f"Search for '{query}' returned {len(search_results)} results.")

        # Send formatted text result back to the server (LLM)
        return ToolResult(result_text.strip(), ToolResultDirection.TO_SERVER)

    except Exception as e:
        logger.exception(f"Error during search execution via RAG provider: {e}")
        # Return a generic error message to the server
        return ToolResult(f"An internal error occurred during search.", ToolResultDirection.TO_SERVER)

async def _execute_grounding(rag_provider: BaseRAGProvider, args: Any) -> ToolResult:
    """Handles the 'report_grounding' tool call by delegating to the RAG provider."""
    source_ids = args.get("sources", [])
    if not isinstance(source_ids, list):
        logger.error(f"Grounding tool called with invalid 'sources' argument type: {type(source_ids)}")
        return ToolResult({"sources": []}, ToolResultDirection.TO_CLIENT) # Send empty list to client

    logger.info(f"Grounding sources requested: {source_ids}")

    if not source_ids:
        # If model requests grounding for empty list, send empty list back
        return ToolResult({"sources": []}, ToolResultDirection.TO_CLIENT)

    try:
        # Delegate detail retrieval to the provider
        detailed_docs: List[Dict[str, Any]] = await rag_provider.get_details(chunk_ids=source_ids)

        # Format the results for the client
        # The provider's get_details should return data in the expected format
        # {"chunk_id": ..., "title": ..., "chunk": ...}
        logger.info(f"Grounding result contains {len(detailed_docs)} documents for client.")

        # Send structured data directly to the client
        return ToolResult({"sources": detailed_docs}, ToolResultDirection.TO_CLIENT)

    except Exception as e:
        logger.exception(f"Error during grounding execution via RAG provider: {e}")
        # Send an empty list to the client in case of error
        return ToolResult({"sources": []}, ToolResultDirection.TO_CLIENT)

def attach_rag_tools(
    rtmt: RTMiddleTier,
    rag_provider: BaseRAGProvider # Accept a provider instance
    ) -> None:
    """
    Attaches RAG tools ('search', 'report_grounding') to the RTMiddleTier,
    using the provided RAG provider for implementation.
    """

    if not isinstance(rag_provider, BaseRAGProvider):
        logger.error(f"Invalid RAG provider passed to attach_rag_tools: {type(rag_provider)}. Tools not attached.")
        return

    # Attach search tool - lambda now calls _execute_search with the provider
    rtmt.tools["search"] = Tool(
        schema=_search_tool_schema,
        target=lambda args: _execute_search(rag_provider, args)
    )

    # Attach grounding tool - lambda now calls _execute_grounding with the provider
    rtmt.tools["report_grounding"] = Tool(
         schema=_grounding_tool_schema,
         target=lambda args: _execute_grounding(rag_provider, args)
    )

    # Log the type of provider attached
    provider_type_name = type(rag_provider).__name__
    logger.info(f"Attached RAG tools using provider: {provider_type_name}")
