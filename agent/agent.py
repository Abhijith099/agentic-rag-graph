"""
Corrective Supervisor Agent for Enterprise RAG.
Orchestrates: Search -> Relevance Grading -> Query Rewriting (if needed) -> Synthesis.
Emits comprehensive RetrievalTrace for explainability and observability.
"""
import os
import logging
from typing import AsyncGenerator, Dict, Any, List, Tuple
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

from agent.config import settings
from agent.prompts import SYSTEM_PROMPT
from agent.schemas import RetrievalTrace, GradeResult, RewrittenQuery
from agent.tools import vector_search, graph_search, hybrid_search
from agent.grader import filter_relevant_chunks
from agent.rewriter import rewrite_query

logger = logging.getLogger(__name__)

# Ensure GROQ_API_KEY is available in the environment for the Groq client
if settings.llm_api_key:
    os.environ["GROQ_API_KEY"] = settings.llm_api_key

# 1. Initialize the primary reasoning model
synthesis_model = GroqModel(settings.llm_choice)

# 2. Primary synthesis agent (generates verified answers grounded in graded context)
synthesis_agent = Agent(
    model=synthesis_model,
    system_prompt=SYSTEM_PROMPT,
    retries=2
)


async def execute_corrective_rag(
    user_query: str
) -> Tuple[str, RetrievalTrace]:
    """
    Executes the full Corrective RAG loop:
    1. Initial retrieval based on query keywords.
    2. Context grading to filter noise.
    3. Self-correction (query rewriting + secondary retrieval) if initial search fails.
    4. Synthesis with citations and full explainability trace.
    
    Returns:
        Tuple of (answer_text, retrieval_trace)
    """
    trace = RetrievalTrace(query=user_query)
    
    # -------------------------------------------------------------
    # STAGE 1: ROUTING & INITIAL RETRIEVAL
    # -------------------------------------------------------------
    query_lower = user_query.lower()
    if any(k in query_lower for k in ["connected", "relationship", "who invested", "partnership", "founder", "ceo"]):
        initial_tool = "graph_search"
        raw_chunks = await graph_search(user_query, limit=5)
    else:
        initial_tool = "vector_search"
        raw_chunks = await vector_search(user_query, limit=3)
    
    trace.tools_invoked.append(initial_tool)
    trace.chunks_retrieved_count = len(raw_chunks)
    
    # -------------------------------------------------------------
    # STAGE 2: CONTEXT RELEVANCE GRADING
    # -------------------------------------------------------------
    relevant_chunks, discarded = await filter_relevant_chunks(user_query, raw_chunks)
    trace.chunks_passed_grading = len(relevant_chunks)
    
    # -------------------------------------------------------------
    # STAGE 3: SELF-HEALING FALLBACK (IF INITIAL SEARCH FAILED)
    # -------------------------------------------------------------
    if not relevant_chunks:
        logger.warning(f"Initial retrieval via {initial_tool} failed. Triggering Corrective RAG fallback...")
        trace.query_was_rewritten = True
        
        # 3a. Ask Rewriter to optimize the query and recommend a new tool
        rewrite_result = await rewrite_query(
            original_query=user_query,
            failure_reason=f"Search using {initial_tool} returned 0 relevant facts."
        )
        trace.rewritten_query = rewrite_result.rewritten_query
        
        # 3b. Execute secondary retrieval using the recommended strategy
        secondary_tool = rewrite_result.search_strategy
        trace.tools_invoked.append(f"{secondary_tool} (retry)")
        
        if secondary_tool == "graph_search":
            retry_chunks = await graph_search(rewrite_result.rewritten_query, limit=5)
        elif secondary_tool == "vector_search":
            retry_chunks = await vector_search(rewrite_result.rewritten_query, limit=3)
        else:
            retry_chunks = await hybrid_search(rewrite_result.rewritten_query, limit=3)
        
        # 3c. Grade the secondary retrieval chunks
        relevant_chunks, _ = await filter_relevant_chunks(
            rewrite_result.rewritten_query, retry_chunks
        )
        trace.chunks_passed_grading = len(relevant_chunks)

    # -------------------------------------------------------------
    # STAGE 4: CITATION EXTRACTION & CONFIDENCE SCORING
    # -------------------------------------------------------------
    citations = set()
    for ch in relevant_chunks:
        src = ch.get("source") or ch.get("document_title") or ch.get("uuid")
        if src:
            citations.add(str(src))
    trace.citations = list(citations)
    
    if len(relevant_chunks) >= 2:
        trace.confidence_level = "high"
    elif len(relevant_chunks) == 1:
        trace.confidence_level = "medium"
    else:
        trace.confidence_level = "none"

    # -------------------------------------------------------------
    # STAGE 5: FINAL SYNTHESIS
    # -------------------------------------------------------------
    if not relevant_chunks:
        answer = (
            "I could not find verified facts regarding this query in the provided knowledge base.\n\n"
            f"*Explainability Note: Both initial search ({initial_tool}) and corrective retry returned zero relevant facts.*"
        )
        return answer, trace

    # Build prompt with verified evidence
    context_text = "\n\n---\n\n".join([
        f"[Source: {ch.get('document_title', ch.get('source', 'Knowledge Graph'))}]\n"
        f"{ch.get('content') or ch.get('fact')}"
        for ch in relevant_chunks
    ])
    
    synthesis_prompt = f"""
USER QUESTION:
{user_query}

VERIFIED RETRIEVED CONTEXT:
{context_text}

Answer the user's question thoroughly based ONLY on the verified context above. Include citations.
""".strip()

    result = await synthesis_agent.run(synthesis_prompt)
    return result.output, trace


async def ask_agent(question: str) -> str:
    """
    Standard non-streaming interface.
    Executes corrective RAG and returns the synthesized answer.
    """
    answer, _ = await execute_corrective_rag(question)
    return answer


async def ask_agent_with_trace(question: str) -> Tuple[str, Dict[str, Any]]:
    """
    Enterprise interface returning both the synthesized answer AND the explainability trace.
    """
    answer, trace = await execute_corrective_rag(question)
    return answer, trace.model_dump()


async def stream_agent(question: str) -> AsyncGenerator[str, None]:
    """
    Streaming query helper.
    Runs corrective RAG and yields the final tokens.
    """
    answer, _ = await execute_corrective_rag(question)
    words = answer.split(" ")
    for word in words:
        yield word + " "
