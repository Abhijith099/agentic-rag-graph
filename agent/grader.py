"""
Context Grader for Corrective RAG (CRAG).
Evaluates retrieved passages for relevance against the user's query,
filtering out noise and detecting retrieval failure.
"""
import logging
from typing import List, Dict, Any, Tuple
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

from agent.config import settings
from agent.schemas import GradeResult
import os

# Ensure GROQ_API_KEY is available for the Groq client
if settings.llm_api_key:
    os.environ["GROQ_API_KEY"] = settings.llm_api_key


logger = logging.getLogger(__name__)

# Specialized system prompt dedicated exclusively to strict factual relevance grading
GRADER_SYSTEM_PROMPT = """
You are a strict, objective Context Relevance Evaluator in an enterprise RAG pipeline.
Your sole job is to determine whether a retrieved text passage contains information 
helpful to answering the user's specific query.

Grading Criteria:
- MARK AS RELEVANT (is_relevant=True): If the passage contains facts, numbers, dates, 
  names, explanations, or context directly related to the user's question.
- MARK AS IRRELEVANT (is_relevant=False): If the passage is merely vaguely on the same broad topic 
  (e.g., general AI buzzwords) but does NOT help answer the specific question asked.

Be strict. Do not assume or extrapolate information that is not explicitly present in the passage.
""".strip()

def create_grader_agent() -> Agent[None, GradeResult]:
    """
    Creates a dedicated Pydantic AI agent configured to output GradeResult.
    Uses result_type=GradeResult to guarantee structured output validation.
    """
    model = GroqModel(settings.llm_choice)
    
    return Agent(
        model=model,
        output_type=GradeResult,    # <--- Forces structured Pydantic output!
        system_prompt=GRADER_SYSTEM_PROMPT,
        retries=2
    )

# Singleton instance of the grader agent
grader_agent = create_grader_agent()

async def grade_chunk(query: str, chunk_content: str) -> GradeResult:
    """
    Evaluates a single chunk against the user's query.
    Returns a GradeResult containing is_relevant, confidence, and reasoning.
    """
    prompt = f"""
USER QUERY:
{query}

RETRIEVED PASSAGE TO EVALUATE:
{chunk_content}

Evaluate whether this passage contains information helpful for answering the query.
""".strip()

    try:
        result = await grader_agent.run(prompt)
        return result.output
    except Exception as e:
        logger.error(f"Grading failed with error: {e}. Defaulting to relevant as fallback.")
        # Graceful fallback: If LLM API hiccups, do not crash; keep chunk with low confidence
        return GradeResult(
            is_relevant=True,
            confidence=0.5,
            reasoning=f"Grading fallback due to API error: {str(e)}"
        )


async def filter_relevant_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    min_confidence: float = 0.6
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Evaluates a list of retrieved chunks in parallel.
    
    Returns:
        Tuple of (relevant_chunks, discarded_chunks)
    """
    if not chunks:
        return [], []

    relevant_chunks: List[Dict[str, Any]] = []
    discarded_chunks: List[Dict[str, Any]] = []

    for chunk in chunks:
        content = chunk.get("content") or chunk.get("fact") or str(chunk)
        grade = await grade_chunk(query, content)
        
        # Attach the grading judgment directly into chunk metadata for explainability!
        chunk_with_grade = dict(chunk)
        chunk_with_grade["grade_reasoning"] = grade.reasoning
        chunk_with_grade["grade_confidence"] = grade.confidence

        if grade.is_relevant and grade.confidence >= min_confidence:
            relevant_chunks.append(chunk_with_grade)
        else:
            discarded_chunks.append(chunk_with_grade)

    logger.info(
        f"Context Grading Complete: {len(relevant_chunks)}/{len(chunks)} chunks deemed relevant."
    )
    return relevant_chunks, discarded_chunks
