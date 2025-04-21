import abc
from typing import List, Dict, Protocol, runtime_checkable

# Using Protocol for structural subtyping (duck typing)
# If you prefer nominal subtyping, you can use abc.ABC and @abc.abstractmethod
@runtime_checkable
class BaseRAGProvider(Protocol):
    """Abstract base class (Protocol) for RAG providers."""

    async def initialize(self) -> None:
        """Initialize the provider (e.g., load data, connect to services)."""
        ... # Default implementation if not needed

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for relevant documents based on the query.

        Args:
            query: The user's search query.
            top_k: The maximum number of results to return.

        Returns:
            A list of dictionaries, each representing a relevant document chunk.
            Expected keys: 'chunk_id', 'text', 'title', 'score'.
        """
        ...

    async def get_details(self, chunk_ids: List[str]) -> List[Dict[str, any]]:
        """
        Retrieve detailed information for specific chunk IDs (for grounding).

        Args:
            chunk_ids: A list of chunk IDs to retrieve details for.

        Returns:
            A list of dictionaries, each containing details for a chunk.
            Expected keys: 'chunk_id', 'title', 'chunk' (text content).
        """
        ... 