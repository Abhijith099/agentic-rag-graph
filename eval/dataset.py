"""
Benchmark Dataset for Enterprise RAG Evaluation.
Contains test cases with questions, expected ground truth answers, and reference documents.
"""
from typing import List
from pydantic import BaseModel, Field  #pyndatic is used for strict data validation and gurantees that every test case has the required fields and correct data types.


class TestCase(BaseModel):
    id: str = Field(..., description="Unique test case identifier") # a short uniq code or label that is prints a result table later , each row has a clead id.
    question: str = Field(..., description="The query to ask the RAG agent") # the prompt we send to rag.
    ground_truth: str = Field(..., description="The factual ground-truth answer") # this is factual ans take directly from neon4j
    expected_doc: str = Field(..., description="The document filename expected to substantiate the answer")


BENCHMARK_DATASET: List[TestCase] = [ #pythone list holds all the test cases, each test case is an instance of TestCase class.
    TestCase(
        id="TC-001",
        question="How much did OpenAI raise in its October 2024 funding round and what was its valuation?",
        ground_truth="OpenAI raised $6.6 billion in its October 2024 funding round at a post-money valuation of $157 billion. Thrive Capital led the round with a $1.2 billion commitment.",
        expected_doc="doc1_openai_funding.md" #the specified markdown file contains the relevant information to answer the question.
    ),
    TestCase(
        id="TC-002",
        question="What is Amazon's total cumulative investment in Anthropic and how much was the November 2024 investment?",
        ground_truth="Amazon's total cumulative investment in Anthropic is $8 billion, which includes an initial $4 billion in September 2023 and an additional $4 billion invested in November 2024.",
        expected_doc="doc2_anthropic_amazon.md" #the specified markdown file contains the relevant information to answer the question.
    ),
    TestCase(
        id="TC-003",
        question="Which Amazon custom AI chips will Anthropic utilize for model training under their expanded partnership?",
        ground_truth="Anthropic will utilize Amazon's Trainium and Inferentia chips for model training and inference.",
        expected_doc="doc2_anthropic_amazon.md"
    ),
    TestCase(
        id="TC-004",
        question="Who was the lead investor in OpenAI's $6.6 billion round in October 2024?",
        ground_truth="Thrive Capital was the lead investor, committing $1.2 billion.",
        expected_doc="doc1_openai_funding.md"
    ),
    TestCase(
        id="TC-005",
        question="What is the official currency used by citizens on planet Mars?",
        ground_truth="Out-of-scope query. The system should state that no verified facts exist in the knowledge base.",
        expected_doc="NONE"
    )
]
