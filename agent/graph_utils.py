"""
Graph utilities for Neo4j and Graphiti integration.
Manages Graphiti client, episode addition, and semantic graph search.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from graphiti_core.driver.neo4j_driver import Neo4jDriver


from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

from agent.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphitiClient:
    """Manages Graphiti knowledge graph operations."""

    def __init__(self):
        self.neo4j_uri = settings.neo4j_uri
        self.neo4j_user = settings.neo4j_user
        self.neo4j_password = settings.neo4j_password

        # LLM setup for Graphiti entity extraction
        self.llm_base_url = settings.llm_base_url
        self.llm_api_key = settings.llm_api_key or "not-needed"
        self.llm_choice = settings.llm_choice

        # Embedding setup for Graphiti node similarity
        self.embedding_base_url = settings.embedding_base_url
        self.embedding_api_key = settings.embedding_api_key or self.llm_api_key
        self.embedding_model = settings.embedding_model
        self.embedding_dimensions = settings.vector_dimension

        self.graphiti: Optional[Graphiti] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initializes Graphiti clients and builds schema indices in Neo4j."""
        if self._initialized:
            return

        logger.info(f"Connecting Graphiti to Neo4j at {self.neo4j_uri}...")

        # 1. LLM client for extraction
        llm_config = LLMConfig(
            api_key=self.llm_api_key,
            model=self.llm_choice,
            small_model=self.llm_choice,
            base_url=self.llm_base_url
        )
        llm_client = OpenAIClient(config=llm_config)

        # 2. Embedder client for graph nodes
        embedder_config = OpenAIEmbedderConfig(
            api_key=self.embedding_api_key,
            embedding_model=self.embedding_model,
            embedding_dim=self.embedding_dimensions,
            base_url=self.embedding_base_url
        )
        embedder = OpenAIEmbedder(config=embedder_config)

        # 3. Reranker client
        reranker = OpenAIRerankerClient(client=llm_client, config=llm_config)
        database_name = getattr(settings, "neo4j_database", None) or os.getenv("NEO4J_DATABASE", self.neo4j_user)
        
        # Create driver explicitly with your custom Aura database name
        graph_driver = Neo4jDriver(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
            database=database_name
        )
        self.graphiti = Graphiti(
            graph_driver=graph_driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=reranker
        )
        


        # Build Graphiti indexes and constraints in Neo4j Aura
        await self.graphiti.build_indices_and_constraints()
        self._initialized = True
        logger.info("Graphiti successfully initialized!")

    async def close(self) -> None:
        """Closes the Graphiti driver on shutdown."""
        if self.graphiti and self._initialized:
            await self.graphiti.close()
            self._initialized = False

    async def add_episode(
        self,
        episode_id: str,
        content: str,
        source: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Adds a document chunk as an Episode into Graphiti."""
        if not self._initialized:
            await self.initialize()

        await self.graphiti.add_episode(
            name=episode_id,
            episode_body=content,
            source=source,
            source_description=f"Source: {source}",
            reference_time=datetime.now(timezone.utc)
        )

    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Searches the knowledge graph using Graphiti's hybrid graph retrieval."""
        if not self._initialized:
            await self.initialize()

        try:
            results = await self.graphiti.search(query)
            return [
                {
                    "fact": getattr(r, "fact", str(r)),
                    "uuid": str(getattr(r, "uuid", "")),
                    "valid_at": str(getattr(r, "valid_at", "")) if hasattr(r, "valid_at") else None
                }
                for r in results[:num_results]
            ]
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    async def clear_graph(self) -> None:
        """Wipes graph data during a clean ingestion run."""
        if not self._initialized:
            await self.initialize()
        await clear_data(self.graphiti.driver)

    # Global Singleton Instance
graph_client = GraphitiClient()


async def initialize_graph() -> None:
    """Initialize the global graph client."""
    await graph_client.initialize()


async def close_graph() -> None:
    """Close the global graph client."""
    await graph_client.close()


async def search_knowledge_graph(query: str) -> List[Dict[str, Any]]:
    """Convenience function to search the graph."""
    return await graph_client.search(query)


async def test_graph_connection() -> bool:
    """Verifies that the graph connection and indices are operational."""
    try:
        await initialize_graph()
        return True
    except Exception as e:
        logger.error(f"Graph connection test failed: {e}")
        return False

