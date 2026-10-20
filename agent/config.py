import os

# Load the database URL from environment variables
db_url = os.environ.get("DATABASE_URL")

db_url = os.environ.get("DATABASE_URL")

"""
Configuration management using pydantic-settings.
Validates environment variables at startup and provides typed access throughout the application.
"""
from typing import Literal
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings validated by Pydantic."""

    # 1. Database Configuration (PostgreSQL + pgvector)
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/agentic_rag",
        description="PostgreSQL connection string with pgvector"
    )

    # 2. Neo4j Configuration (Knowledge Graph)
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt protocol URI"
    )
    neo4j_password: str = Field(
        default="password",
        description="Neo4j database password"
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username"
    )
    neo4j_username: str | None = Field(
        default=None,
        description="Alias for neo4j_user"
    )

    @property
    def effective_neo4j_user(self) -> str:
        return self.neo4j_username or self.neo4j_user


    neo4j_password: str = Field(
        default="password",
        description="Neo4j database password"
    )

    # 3. LLM Configuration (The Agent Brain - Supports OpenAI, Groq, Ollama, etc.)
    llm_provider: str = Field(
        default="openai",
        description="Provider identifier (openai, ollama, groq, etc.)"
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the LLM API (e.g. Groq or OpenAI)"
    )
    llm_api_key: str | None = Field(
        default=None,
        description="API key for the LLM provider"
    )
    llm_choice: str = Field(
        default="llama-3.1-8b-instant",
        description="Model name to use for agent reasoning"
    )
    ingestion_llm_choice: str | None = Field(
        default="llama-3.1-8b-instant",
        description="Model name for document ingestion/extraction"
    )

    # 4. Embedding Configuration (The Vector Creator)
    embedding_provider: str = Field(
        default="openai",
        description="Provider used to generate vector embeddings"
    )
    embedding_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for embedding API"
    )
    embedding_api_key: str | None = Field(
        default=None,
        description="API key for embedding provider"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model name for embeddings"
    )
    vector_dimension: int = Field(
        default=1536,
        description="Vector dimension (must match vector(N) in sql/schema.sql)"
    )

    # 5. Ingestion & Chunking Parameters
    chunk_size: int = Field(
        default=800,
        description="Target token/character count per chunk"
    )
    chunk_overlap: int = Field(
        default=150,
        description="Overlap between consecutive chunks to preserve context"
    )
    max_chunk_size: int = Field(
        default=1500,
        description="Hard maximum chunk size"
    )

    # 6. Server & App Settings
    app_env: str = Field(default="development", description="Environment mode")
    log_level: str = Field(default="INFO", description="Logging verbosity")
    app_host: str = Field(default="0.0.0.0", description="Host interface to bind")
    app_port: int = Field(default=8058, description="Port for the FastAPI server")

    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # Ignore extra fields in .env without erroring
        case_sensitive=False     # Case-insensitive matching: LLM_API_KEY -> llm_api_key
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached instance of Settings.
    lru_cache ensures .env is parsed only once in memory.
    """
    return Settings()


# Global convenience instance
settings = get_settings()
