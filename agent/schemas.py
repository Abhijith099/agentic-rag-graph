"""
Data contracts and Pydantic schemas for the Corrective RAG pipeline.
Defines structured outputs for grading, query rewriting, and explainability traces.
"""
from typing import List, Literal, Dict, Any, Optional
from pydantic import BaseModel, Field

from agent.tools import graph_search # base model is the foundation of pydantic

class GradeResult(BaseModel):
    """
    Structred judgment on wheather a retrieved documents chunk is relevaant tot he users;s question."""
    is_relevant: bool = Field(..., description="Whether the retrieved document chunk is relevant to the user's question.")
    confidence: float = Field(..., 
        ge=0.0, #greater than or equal to 0.0
        le=1.0,#less than or equal to 1.0
        description="Confidence score of the relevance judgment.")
    reasoning:str = Field(..., description="Explanation of the reasoning behind the relevance judgment.")
    # Field(......) this means it is mandatory, and the description is for documentation purposes.

class RewrittenQuery(BaseModel):
    """ 
    Structured output when the agend decides an initial query failded and needs to eb refreamed for better results.
    """
    orginal_query: str = Field(..., description="The original user query that was deemed insufficient.")
    rewritten_query: str = Field(..., description="The rewritten query that is expected to yield better results.")
    search_strategy: Literal["vector_search","graph_search", "hybrid_search"] = Field(..., description="The strategy used to rewrite the query, indicating whether it was expanded, refined, or clarified.")
    # the literal chooses only those option which are specified in the list if there were literal then it would have been a string and any string could have been passed in the search_strategy field.
    rationale: str = Field(..., description="Explanation of why the query was rewritten and how the new query is expected to improve results.")
    #if the vector search fails then it will try for graph rationale: "Original vector query returned no passages; switching to graph search to traverse investment relationships."
    #rewritten_query: "Amazon investment and cloud partnership with Anthropic"
    #search_strategy: "graph_search"
    #rationale: "Original vector query returned no passages; switching to graph search to traverse investment relationships."


class RetrievalTrace(BaseModel):
    """
    Detailed audit log tracing how an answer was derived.
    Provides compelete transparency into aget decisions for enterprise observability.
    """
    query: str = Field(..., description="The query processed by the agent.")
    tools_invoked: List[str] = Field(default_factory=list, description="List of tools called during execution.")
    chunks_retrieved_count: int = Field(default=0, description="Total chunks fetched before grading.")
    chunks_passed_grading: int = Field(default=0, description="Number of chunks that passed relevance grading.")
    query_was_rewritten: bool = Field(default=False, description="Whether the query had to be rewritten.")
    rewritten_query: Optional[str] = Field(default=None, description="The rewritten query if rewriting occurred.")
    citations: List[str] = Field(default_factory=list, description="List of document titles or sources cited.")
    confidence_level: Literal["high", "medium", "low", "none"] = Field(
        default="none",
        description="Overall confidence in the final synthesized answer."
    )
    #Which tools were invoked?
    #How many chunks were evaluated, and how many were discarded as noise?
    #Did the agent have to rewrite the user's question?
    #What is the confidence level?