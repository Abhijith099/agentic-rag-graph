# 🧠 Agentic RAG with Knowledge Graph

> An enterprise-grade, dual-layer Agentic Retrieval-Augmented Generation (RAG) system combining **PostgreSQL (pgvector)** semantic search with a **Neo4j Knowledge Graph** orchestrated by **Pydantic AI**.

---

## 📌 Architecture Overview

Traditional RAG systems rely solely on vector search, which retrieves document passages based on semantic similarity but struggles with complex, multi-hop entity relationships (e.g., *Which company funded by former OpenAI researchers competes with Google?*).

This project integrates two complementary retrieval engines into an autonomous agent loop:

`
                            ┌────────────────────────┐
                            │      User Request      │
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │   Pydantic AI Agent    │
                            │ (Groq Llama 3.3 / GPT) │
                            └───────────┬────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
   ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
   │   vector_search   │      │   graph_search    │      │   hybrid_search   │
   ├───────────────────┤      ├───────────────────┤      ├───────────────────┤
   │ PostgreSQL + HNSW │      │ Neo4j + Graphiti  │      │ Vector + Keyword  │
   │ Technical docs,   │      │ Entities, nodes,  │      │ Exact model codes │
   │ narrative context │      │ multi-hop links   │      │ & product names   │
   └───────────────────┘      └───────────────────┘      └───────────────────┘
`

---

## ✨ Key Features

- **Dual-Layer Retrieval**:
  - **Vector DB (Neon PostgreSQL)**: Stores chunked passages and dense vector embeddings with an HNSW cosine distance index.
  - **Knowledge Graph (Neo4j Aura)**: Stores entities and relations with temporal memory via **Graphiti**.
- **Autonomous Tool Selection**: The **Pydantic AI** agent dynamically chooses between ector_search, graph_search, or hybrid_search based on the user's intent.
- **Semantic Text Chunking**: Intelligent recursive chunker splitting on paragraph and sentence boundaries with configurable context overlap.
- **Real-Time Streaming API**: Built on **FastAPI** with **Server-Sent Events (SSE)** for instant token streaming.
- **Rich Interactive CLI**: Terminal interface with syntax highlighting, live token typing, and tool execution status.

---

## 📁 Repository Structure

`
├── agent/
│   ├── config.py         # Type-safe settings via pydantic-settings
│   ├── db.py             # Async connection pool (asyncpg) to PostgreSQL
│   ├── graph_utils.py    # Neo4j Aura & Graphiti integration
│   ├── tools.py          # vector_search, graph_search, hybrid_search
│   ├── prompts.py        # System prompt with decision rules
│   ├── agent.py          # Pydantic AI agent definition
│   └── api.py            # FastAPI REST and SSE streaming server
├── ingestion/
│   ├── chunker.py        # Semantic chunking with overlap
│   ├── embedder.py       # Batch vector embedding generator
│   └── ingest.py         # Full document ingestion pipeline
├── sql/
│   └── schema.sql        # pgvector extension, HNSW indexes & stored functions
├── big_tech_docs/        # Sample markdown documents for ingestion
├── cli.py                # Interactive streaming terminal client
├── .env.example          # Environment variable template
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
`

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- A free [Neon.tech](https://neon.tech) PostgreSQL instance (with pgvector)
- A free [Neo4j Aura](https://neo4j.com/cloud/aura/) instance
- A [Groq](https://console.groq.com) API key (or OpenAI API key)
- Local [Ollama](https://ollama.ai) (optional, for local embeddings)

### 2. Installation
Clone the repository and set up a virtual environment:

`ash
git clone https://github.com/your-username/agentic-rag-knowledge-graph.git
cd agentic-rag-knowledge-graph

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
`

### 3. Environment Configuration
Copy .env.example to .env and fill in your credentials:

`ash
cp .env.example .env
`

### 4. Initialize Database Schema
Run the database initializer to create the pgvector tables and HNSW index:

`ash
python -c import asyncio; from agent.db import init_db; asyncio.run(init_db())
`

### 5. Ingest Documents
Process the markdown files into vector embeddings and knowledge graph entities:

`ash
python -m ingestion.ingest -d big_tech_docs --clean --verbose
`

---

## 💻 Running the Application

### Step 1: Start the API Server
In Terminal 1, start the FastAPI streaming server:

`ash
python -m agent.api
`
The server will start on http://localhost:8058. You can view the interactive Swagger API documentation at http://localhost:8058/docs.

### Step 2: Launch the Interactive CLI
In Terminal 2, launch the streaming terminal interface:

`ash
python cli.py
`

---

## 🧪 Example Queries & Behavior

- **Relationship Query**:
  > *How is Microsoft connected to OpenAI?*  
  > ➜ **Tool Used**: graph_search (traverses partnerships, investments, and Azure cloud exclusivity in Neo4j).
- **Technical Narrative Query**:
  > *Explain Google's TPU architecture and multimodal training.*  
  > ➜ **Tool Used**: ector_search (retrieves technical descriptions from PostgreSQL).
- **Exact Keyword / Model Query**:
  > *What are the specifications of Llama-3.3-70b-versatile?*  
  > ➜ **Tool Used**: hybrid_search (combines trigram keyword matching with cosine similarity).

---

## 📄 License
This project is licensed under the MIT License.
