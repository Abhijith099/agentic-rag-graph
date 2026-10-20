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
                query_vector, query, limit, text_weight
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
