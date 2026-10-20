"""
Pydantic AI Agent implementation.
Integrates Groq LLM with vector_search, graph_search, and hybrid_search tools.
Supports streaming responses and autonomous tool execution.
"""
import os
import logging
from typing import AsyncGenerator
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

from agent.config import settings
from agent.prompts import SYSTEM_PROMPT
from agent.tools import vector_search, graph_search, hybrid_search

logger = logging.getLogger(__name__)

# Ensure GROQ_API_KEY is available in the environment for the Groq client
if settings.llm_api_key:
    os.environ["GROQ_API_KEY"] = settings.llm_api_key

# 1. Initialize the Groq Model (e.g. llama-3.1-8b-instant)
model = GroqModel(settings.llm_choice)

# 2. Instantiate the Pydantic AI Agent with our System Prompt
rag_agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    retries=2
)


# 3. Register our 3 standalone tools with the Agent
rag_agent.tool_plain(vector_search)
rag_agent.tool_plain(graph_search)
rag_agent.tool_plain(hybrid_search)



async def ask_agent(question: str) -> str:
    """
    Non-streaming query helper.
    Runs the agent and returns the complete synthesized answer.
    """
    result = await rag_agent.run(question)
    return result.output   # <--- change 'data' to 'output'



async def stream_agent(question: str) -> AsyncGenerator[str, None]:
    """
    Streaming query helper.
    Yields token chunks in real time as the LLM generates them.
    """
    async with rag_agent.run_stream(question) as result:
        async for chunk in result.stream_text(delta=True):
            yield chunk
