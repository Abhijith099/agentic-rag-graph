"""
Embedding generation service.
Supports OpenAI API, OpenAI-compatible providers, and Ollama.
Converts text chunks into high-dimensional vector representations.
"""
from typing import List
from openai import AsyncOpenAI
from agent.config import settings
from ingestion.chunker import TextChunk


class Embedder:
    """Handles vector embedding generation for text chunks."""

    def __init__(self):
        # Configure the AsyncOpenAI client
        # If using Ollama, base_url is http://localhost:11434/v1 and api_key is 'ollama'
        api_key = settings.embedding_api_key or settings.llm_api_key or "not-needed"
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=settings.embedding_base_url
        )
        self.model = settings.embedding_model
        self.dimensions = settings.vector_dimension

    async def embed_text(self, text: str) -> List[float]:
        """
        Generates an embedding vector for a single query string.
        Used at search time when a user asks a question.
        """
        clean_text = text.replace("\n", " ").strip()
        response = await self.client.embeddings.create(
            input=[clean_text],
            model=self.model
        )
        # Extract the float list from the response object
        return response.data[0].embedding

    async def embed_chunks(self, chunks: List[TextChunk], batch_size: int = 50) -> List[List[float]]:
        """
        Generates embeddings for a list of TextChunks in batches.
        Batching reduces network round-trips and stays within API rate limits.
        """
        if not chunks:
            return []

        all_embeddings: List[List[float]] = []
        texts = [chunk.content.replace("\n", " ").strip() for chunk in chunks]

        # Process in batches of 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

