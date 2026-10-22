"""
Query Rewriter for Corrective RAG (CRAG).
Analyzes ambiguous, colloquial, or failed user queries and reformulates them
into optimized search queries while selecting the most effective retrieval strategy.
"""
import os
import logging
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

from agent.config import settings
from agent.schemas import RewrittenQuery

logger = logging.getLogger(__name__)

# Ensure GROQ_API_KEY is available
if settings.llm_api_key:
    os.environ["GROQ_API_KEY"] = settings.llm_api_key

REWRITER_SYSTEM_PROMPT = """
You are an expert Query Reformulation and Search Strategist in an enterprise RAG pipeline.
Your job is to rewrite suboptimal user questions into crisp, keyword-dense search queries 
and choose the best retrieval tool.

Available Search Tools:
- 'graph_search': Use when the question involves entity relationships, leadership connections, 
  investments, acquisitions, partnerships, or multi-hop connections.
- 'vector_search': Use when the question requires descriptive technical narratives, 
  architecture overviews, definitions, or broad thematic explanations.
- 'hybrid_search': Use when the query contains exact model names (e.g., GPT-4o), 
  product codes, error IDs, or specific terminology alongside semantic meaning.

Rules:
1. Strip all conversational filler ("Can you tell me", "I was wondering if", etc.).
2. Resolve pronouns (replace "they", "it", "the company" with explicit company/person names).
3. Select the single most effective tool in search_strategy.
4. Explain your rationale clearly.
""".strip()


def create_rewriter_agent() -> Agent[None, RewrittenQuery]:
    """
    Creates a dedicated Pydantic AI agent configured to output RewrittenQuery.
    Uses output_type=RewrittenQuery to enforce structured outputs.
    """
    model = GroqModel(settings.llm_choice)
    
    return Agent(
        model=model,
        output_type=RewrittenQuery,
        system_prompt=REWRITER_SYSTEM_PROMPT,
        retries=2
    )

# Singleton instance of the rewriter agent
rewriter_agent = create_rewriter_agent()


async def rewrite_query(
    original_query: str,
    failure_reason: str = "Initial retrieval returned insufficient or irrelevant results."
) -> RewrittenQuery:
    """
    Rewrites a query that failed or yielded weak context during retrieval.
    
    Args:
        original_query: The user's original query.
        failure_reason: The context or error explaining why the previous search failed.
        
    Returns:
        A structured RewrittenQuery with rewritten text and recommended search strategy.
    """
    prompt = f"""
ORIGINAL USER QUERY:
{original_query}

REASON FOR REWRITE:
{failure_reason}

Reformulate this query for maximum retrieval recall and select the optimal search strategy.
""".strip()

    try:
        result = await rewriter_agent.run(prompt)
        logger.info(
            f"Query Rewritten: '{original_query}' -> '{result.output.rewritten_query}' "
            f"using {result.output.search_strategy}"
        )
        return result.output
    except Exception as e:
        logger.error(f"Query rewriting failed with error: {e}. Falling back to original query.")
        # Graceful fallback: return original query with hybrid_search
        return RewrittenQuery(
            original_query=original_query,
            rewritten_query=original_query,
            search_strategy="hybrid_search",
            rationale=f"Fallback due to rewriter error: {str(e)}"
        )
