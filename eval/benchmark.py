"""
Automated Evaluation Suite for Corrective RAG.
Evaluates the RAG Triad (Context Relevance, Faithfulness, Answer Relevance) using an LLM Judge.
Generates an executive summary benchmark report.
"""
import os
import asyncio
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from agent.config import settings
from agent.agent import execute_corrective_rag
from eval.dataset import BENCHMARK_DATASET, TestCase

# Silence noisy third-party database logs during evaluation
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

console = Console()

# -------------------------------------------------------------
# 1. EVALUATION DATA CONTRACT (RAG TRIAD SCHEMA)
# -------------------------------------------------------------
class EvaluationScore(BaseModel):
    """Structured evaluation scores produced by the LLM Judge."""
    context_relevance: float = Field(
        ..., ge=0.0, le=1.0,
        description="1.0 if retrieved chunks contain facts answering the question; 0.0 if completely irrelevant."
    )
    faithfulness: float = Field(
        ..., ge=0.0, le=1.0,
        description="1.0 if all claims in the generated answer are supported by retrieved context; 0.0 if hallucinated."
    )
    answer_relevance: float = Field(
        ..., ge=0.0, le=1.0,
        description="1.0 if the answer directly and completely addresses the user question; 0.0 if not."
    )
    reasoning: str = Field(..., description="Brief 1-sentence justification for the scores.")

# -------------------------------------------------------------
# 2. LLM JUDGE AGENT SETUP
# -------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """
You are an impartial, strict evaluation judge for Enterprise RAG systems.
Your job is to evaluate candidate RAG answers against:
1. USER QUESTION
2. RETRIEVED CONTEXT
3. GENERATED ANSWER
4. GROUND TRUTH ANSWER

Scoring Rules:
- context_relevance (0.0 to 1.0): Did the retrieved context contain facts that answer the question?
  * Note: For intentional out-of-scope/trap questions where ground truth is 'Out-of-scope query', if 0 chunks passed grading or empty context was safely kept, award 1.0.
- faithfulness (0.0 to 1.0): Are all statements in the answer strictly grounded in the context without fabricating unverified facts?
  * If the system safely states that no verified facts were found, and context has 0 facts, faithfulness is 1.0.
- answer_relevance (0.0 to 1.0): Does the answer directly address what the user asked?

Be strict, objective, and precise.
""".strip()

judge_model = GroqModel(settings.llm_choice)
judge_agent = Agent(
    model=judge_model,
    system_prompt=JUDGE_SYSTEM_PROMPT,
    output_type=EvaluationScore,
    retries=2
)
# -------------------------------------------------------------
# 3. SINGLE TEST CASE EVALUATOR
# -------------------------------------------------------------
async def evaluate_single_case(test_case: TestCase) -> Dict[str, Any]:
    """
    Runs a single test case through Corrective RAG, then grades it with the Judge Agent.
    """
    console.print(f"[bold cyan]Running {test_case.id}:[/bold cyan] [italic]{test_case.question}[/italic]")
    
    # 1. Execute Corrective RAG
    answer, trace = await execute_corrective_rag(test_case.question)
    
    # 2. Format context for Judge
    context_summary = f"Tools: {trace.tools_invoked}, Chunks passed: {trace.chunks_passed_grading}, Rewritten: {trace.query_was_rewritten}"
    
    judge_prompt = f"""
USER QUESTION:
{test_case.question}

GROUND TRUTH ANSWER:
{test_case.ground_truth}

RETRIEVED CONTEXT SUMMARY:
{context_summary}

GENERATED ANSWER:
{answer}
""".strip()

    # 3. Grade using the LLM Judge
    judge_result = await judge_agent.run(judge_prompt)
    score: EvaluationScore = judge_result.output
    
    return {
        "id": test_case.id,
        "question": test_case.question,
        "answer": answer,
        "trace": trace,
        "score": score
    }
# -------------------------------------------------------------
# 4. BENCHMARK RUNNER & REPORT GENERATOR
# -------------------------------------------------------------
async def run_benchmark():
    """
    Executes the complete benchmark suite across all dataset test cases.
    Computes averages and displays a comprehensive Rich report.
    """
    console.print(
        Panel.fit(
            "[bold green]Enterprise Corrective RAG Evaluation Suite[/bold green]\n"
            "[dim]Benchmarking Context Relevance, Faithfulness & Answer Relevance[/dim]",
            border_style="green"
        )
    )
    
    results = []
    for tc in BENCHMARK_DATASET:
        try:
            res = await evaluate_single_case(tc)
            results.append(res)
            # Short pause to prevent API rate limits
            await asyncio.sleep(0.5)
        except Exception as e:
            console.print(f"[bold red]Failed {tc.id}: {e}[/bold red]")

    # -------------------------------------------------------------
    # RENDER EXECUTIVE REPORT TABLE
    # -------------------------------------------------------------
    table = Table(title="RAG Triad Benchmark Results", border_style="cyan", show_lines=True)
    table.add_column("Test ID", style="bold", justify="center")
    table.add_column("Context Rel.", justify="center")
    table.add_column("Faithfulness", justify="center")
    table.add_column("Answer Rel.", justify="center")
    table.add_column("Self-Healed?", justify="center")
    table.add_column("Confidence", justify="center")
    table.add_column("Judge Reasoning", style="dim")

    total_cr = 0.0
    total_f = 0.0
    total_ar = 0.0

    for r in results:
        sc: EvaluationScore = r["score"]
        tr = r["trace"]
        
        total_cr += sc.context_relevance
        total_f += sc.faithfulness
        total_ar += sc.answer_relevance
        
        cr_col = "green" if sc.context_relevance >= 0.8 else "yellow" if sc.context_relevance >= 0.5 else "red"
        f_col = "green" if sc.faithfulness >= 0.8 else "yellow" if sc.faithfulness >= 0.5 else "red"
        ar_col = "green" if sc.answer_relevance >= 0.8 else "yellow" if sc.answer_relevance >= 0.5 else "red"
        
        rewritten_str = "[yellow]Yes[/yellow]" if tr.query_was_rewritten else "[dim]No[/dim]"
        conf_str = tr.confidence_level.upper()
        
        table.add_row(
            r["id"],
            f"[{cr_col}]{sc.context_relevance:.2f}[/{cr_col}]",
            f"[{f_col}]{sc.faithfulness:.2f}[/{f_col}]",
            f"[{ar_col}]{sc.answer_relevance:.2f}[/{ar_col}]",
            rewritten_str,
            conf_str,
            sc.reasoning
        )

    console.print()
    console.print(table)

    count = len(results) or 1
    avg_cr = total_cr / count
    avg_f = total_f / count
    avg_ar = total_ar / count
    rag_triad_score = (avg_cr + avg_f + avg_ar) / 3.0

    summary_panel = (
        f"[bold]Average Context Relevance:[/bold]   {avg_cr:.2%}\n"
        f"[bold]Average Faithfulness:[/bold]          {avg_f:.2%}\n"
        f"[bold]Average Answer Relevance:[/bold]      {avg_ar:.2%}\n"
        f"[bold green]Composite RAG Triad Score:[/bold green]     {rag_triad_score:.2%}"
    )
    console.print(Panel(summary_panel, title="[bold green]Executive Benchmark Summary[/bold green]", border_style="green"))


if __name__ == "__main__":
    asyncio.run(run_benchmark())
