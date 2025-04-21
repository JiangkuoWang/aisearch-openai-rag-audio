import logging
import numpy as np
import openai
import asyncio
import time
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRAGProvider
# ToolResult/Direction might not be needed directly here if attach_rag_tools handles it
# from rtmt import ToolResult, ToolResultDirection

logger = logging.getLogger(__name__)

class InMemoryRAGProvider(BaseRAGProvider):
    """RAG provider using in-memory data (JSONL metadata, NumPy vectors)."""

    def __init__(self,
                 openai_client: openai.OpenAI,
                 embedding_model: str,
                 all_metadata: List[Dict[str, Any]],
                 all_vectors: np.ndarray):

        if not all_metadata or all_vectors is None or all_vectors.size == 0:
            raise ValueError("Cannot initialize InMemoryRAGProvider with empty metadata or vectors.")

        self.openai_client = openai_client
        self.embedding_model = embedding_model
        self.all_metadata = all_metadata
        self.all_vectors = all_vectors
        self.metadata_map = {item['chunk_id']: item for item in self.all_metadata}
        logger.info(f"Initialized InMemoryRAGProvider with {len(self.all_metadata)} items.")

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs cosine similarity search against in-memory vectors."""
        search_start_time = time.perf_counter() # Start total timer

        if not query:
            logger.error("Search called without a query.")
            return [] # Return empty list for no query

        logger.info(f"Searching in-memory for '{query}' (top_k={top_k})...")

        try:
            # 1. Get query embedding
            embed_api_start_time = time.perf_counter() # Start API timer
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, # Use default executor
                lambda: self.openai_client.embeddings.create(
                    input=[query],
                    model=self.embedding_model
                )
            )
            query_vector = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
            embed_api_end_time = time.perf_counter() # End API timer
            embed_api_latency = (embed_api_end_time - embed_api_start_time) * 1000 # milliseconds
            # Log API latency
            logger.info(f"[Search Timing] Embedding API call took: {embed_api_latency:.2f} ms")


            # 2. Calculate cosine similarities & Sort
            similarity_start_time = time.perf_counter() # Start search timer
            if query_vector.shape[1] != self.all_vectors.shape[1]:
                 logger.error(f"Query vector dimension ({query_vector.shape[1]}) does not match index dimension ({self.all_vectors.shape[1]}).")
                 # Optionally raise an error or return empty list
                 return []

            similarities = cosine_similarity(query_vector, self.all_vectors)[0]

            # 3. Get top K results
            actual_k = min(top_k, len(similarities))
            if actual_k == 0:
                 logger.info("No results found in the knowledge base.")
                 return []

            # Get indices and scores of top K
            top_k_indices = np.argsort(similarities)[-actual_k:][::-1] # [::-1] reverses to descending order
            similarity_end_time = time.perf_counter() # End search timer
            similarity_latency = (similarity_end_time - similarity_start_time) * 1000 # milliseconds
            # Log search latency
            logger.info(f"[Search Timing] Local similarity search (calc + sort) took: {similarity_latency:.2f} ms")


            # 4. Format results according to BaseRAGProvider spec
            formatting_start_time = time.perf_counter() # Start formatting timer
            results = []
            for i in top_k_indices:
                metadata = self.all_metadata[i]
                score = float(similarities[i]) # Ensure score is a standard float
                results.append({
                    "chunk_id": metadata.get("chunk_id", f"missing-id-{i}"),
                    "text": metadata.get("text", ""),
                    "title": metadata.get("title", "Unknown"),
                    "score": score
                })
                # logger.info(f"  Found: {metadata.get('chunk_id', 'N/A')} (Score: {score:.4f})") # Optional: Less verbose during timing
            formatting_end_time = time.perf_counter() # End formatting timer
            formatting_latency = (formatting_end_time - formatting_start_time) * 1000 # milliseconds
            # Log formatting latency
            logger.info(f"[Search Timing] Result formatting took: {formatting_latency:.2f} ms")

            search_end_time = time.perf_counter() # End total timer
            total_search_latency = (search_end_time - search_start_time) * 1000 # milliseconds
            # Log total search method latency
            logger.info(f"[Search Timing] Total 'search' execution took: {total_search_latency:.2f} ms for query: '{query[:50]}...'")

            return results

        except openai.APIError as e:
            logger.error(f"OpenAI API error during embedding query: {e}")
            # Log timing even on error if possible
            search_end_time = time.perf_counter()
            total_search_latency = (search_end_time - search_start_time) * 1000
            logger.error(f"[Search Timing] OpenAI API Error occurred after {total_search_latency:.2f} ms in 'search'.")
            return []
        except Exception as e:
            logger.exception(f"Error during in-memory search: {e}")
            # Log timing even on error if possible
            search_end_time = time.perf_counter()
            total_search_latency = (search_end_time - search_start_time) * 1000
            logger.error(f"[Search Timing] Error occurred after {total_search_latency:.2f} ms during 'search' execution.")
            return []

    async def get_details(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieves details for given chunk IDs from the in-memory metadata."""
        details_start_time = time.perf_counter() # Start timer

        # Reduced log noise for the main operation
        # logger.info(f"Getting details for {len(chunk_ids)} chunk IDs: {chunk_ids}")
        docs = []
        found_ids = set()

        for source_id in chunk_ids:
            if source_id in self.metadata_map and source_id not in found_ids:
                item = self.metadata_map[source_id]
                # Format according to BaseRAGProvider spec
                docs.append({
                    "chunk_id": item.get("chunk_id", source_id),
                    "title": item.get("title", "Unknown"),
                    "chunk": item.get("text", "") # Changed key from 'text' to 'chunk'
                })
                found_ids.add(source_id)
            # else: # Reduced log noise for timing
            #      if source_id in found_ids:
            #          logger.debug(f"  Duplicate grounding source ID requested: {source_id}")
            #      else:
            #          logger.warning(f"  Grounding source ID not found in metadata: {source_id}")

        details_end_time = time.perf_counter() # End timer
        total_details_latency = (details_end_time - details_start_time) * 1000 # milliseconds
        # Log total get_details latency
        logger.info(f"[Details Timing] Total 'get_details' execution took: {total_details_latency:.2f} ms for {len(chunk_ids)} IDs")

        return docs