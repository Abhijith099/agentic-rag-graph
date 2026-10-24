"""
Tools exposed to the Pydantic AI Agent.
Includes vector_search (Postgres pgvector), graph_search (Neo4j Graphiti),
and hybrid_search (combining text and vector retrieval).
"""
import logging
from typing import List, Dict, Any, Optional

from agent.db import get_connection
from agent.graph_utils import search_knowledge_graph
from ingestion.embedder import Embedder

logger = logging.getLogger(__name__)

# Singleton embedder instance for query embeddings
_embedder = Embedder()


async def vector_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search the PostgreSQL vector database for document chunks semantically similar to the query.
    
    BEST FOR:
    - Topic-based questions (e.g., 'What is Google's multimodal architecture?')
    - Detailed technical descriptions, summaries, and narrative explanations.
    - Answering questions about specific document contents.
    
    Args:
        query: The semantic search query string.
        limit: Maximum number of relevant chunks to retrieve (default: 5).
        
    Returns:
        A list of matching chunks with content, similarity score, and document source.
    """
    try:
        query_vector = await _embedder.embed_text(query)
        vector_str = str(query_vector)
        async with get_connection() as conn:
            records = await conn.fetch(
                """
                SELECT 
                    chunk_id,
                    content,
                    similarity,
                    document_title,
                    document_source
                FROM match_chunks($1::vector, $2);
                """,
                vector_str, limit 
            )
            return [
                {
                    "content": r["content"],
                    "similarity": round(float(r["similarity"]), 4),
                    "document_title": r["document_title"],
                    "source": r["document_source"]
                }
                for r in records
            ]
    except Exception as e:
        logger.error(f"vector_search failed: {e}")
        return [{"error": f"Vector search error: {str(e)}"}]


async def graph_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search the Neo4j Knowledge Graph for entity relationships, facts, and connections.
    
    BEST FOR:
    - Questions about how entities connect (e.g., 'Who invested in OpenAI?')
    - Multi-hop relationship queries (e.g., 'Which companies founded by ex-Googlers received funding?')
    - Discovering partnerships, acquisitions, leadership, and competitor networks.
    
    Args:
        query: Search query focusing on entities and relationships.
        limit: Maximum number of relationship facts to retrieve (default: 10).
        
    Returns:
        A list of relationship facts and entities from the graph.
    """
    try:
        facts = await search_knowledge_graph(query)
        return facts[:limit]
    except Exception as e:
        logger.error(f"graph_search failed: {e}")
        return [{"error": f"Graph search error: {str(e)}"}]


async def hybrid_search(query: str, limit: int = 5, text_weight: float = 0.3) -> List[Dict[str, Any]]:
    """
    Combines semantic vector search with keyword/trigram full-text search.
    
    BEST FOR:
    - Queries containing specific keywords, error codes, model names (e.g., 'GPT-4o', 'Llama-3.1-8b')
    - When pure vector search might miss exact keyword matches.
    
    Args:
        query: Search query text.
        limit: Number of results to return.
        text_weight: Weight given to keyword matching (0.0 to 1.0, default: 0.3).
    """
    try:
        query_vector = await _embedder.embed_text(query)
        vector_str = str(query_vector)
        async with get_connection() as conn:
            records = await conn.fetch(
                """
                SELECT 
                    chunk_id,
                    content,
                    combined_score,
                    vector_similarity,
                    text_similarity,
                    document_title,
                    document_source
                FROM hybrid_search($1::vector, $2, $3, $4);
                """,
                vector_str, query, limit, text_weight
            )
            return [
                {
                    "content": r["content"],
                    "combined_score": round(float(r["combined_score"]), 4),
                    "vector_similarity": round(float(r["vector_similarity"]), 4),
                    "text_similarity": round(float(r["text_similarity"]), 4),
                    "document_title": r["document_title"],
                    "source": r["document_source"]
                }
                for r in records
            ]
    except Exception as e:
        logger.error(f"hybrid_search failed: {e}")
        return [{"error": f"Hybrid search error: {str(e)}"}]
