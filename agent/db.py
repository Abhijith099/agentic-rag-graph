"""Database connection and session management for PostgreSQL with pgvector support.
Use asyncpg for high-perfomance asynchronous query execution."""
import os
import asyncpg  #fastest async postgresql dirver in python
from contextlib import asynccontextmanager #allows python  async with get_db_connection()
from typing import AsyncGenerator
from agent.config import Settings #imports config

# Global reference to the connection pool
_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """
    Initializes and returns the shared asyncpg connection pool.
    Reuses existing pool if already created (Singleton pattern).
    """
    global _pool
    if _pool is None:
        # Convert SQLAlchemy-style URLs (postgresql+asyncpg://) to raw asyncpg (postgresql://) if needed
        db_url = Settings().database_url.replace("postgresql+asyncpg://", "postgresql://")
        
        _pool = await asyncpg.create_pool(
            dsn=db_url,
            min_size=2,          # Keep at least 2 connections ready
            max_size=10,         # Scale up to 10 connections under load
            command_timeout=60   # Vector searches can take time on large datasets
        )
    return _pool


async def close_pool() -> None:
    """Gracefully closes all connections in the pool on server shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """ 
    Context manager that yields an active connection from the pool.
    Usage:
        async with get_connection() as conn:
            results = await conn.fetch("SELECT * FROM documents;")
    """
    pool = await get_pool()
    async with pool.acquire() as connection:
        yield connection

# Alias so both get_connection and get_db_connection work!
get_db_connection = get_connection


async def check_database_health() -> bool:
    """
    Checks if PostgreSQL is accessible and pgvector extension is installed.
    Returns True if healthy, returns False otherwise.
    """
    try:
        async with get_connection() as conn:
            # Check basic connectivity
            result = await conn.fetchval("SELECT 1;")
            if result != 1:
                return False
            # Check if pgvector extension is loaded
            vector_check = await conn.fetchval(
                "SELECT count(*) FROM pg_extension WHERE extname = 'vector';"
            )
            return vector_check > 0
    except Exception as e:
        print(f"Database health check failed: {e}")
        return False


async def init_db(schema_path: str = "sql/schema.sql") -> None:
    """
    Initializes the database by executing the schema SQL file.
    This should be called once during application startup.
    """
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read() 
    print(f"Applying schema from {schema_path}...")
    async with get_connection() as conn:
        await conn.execute(schema_sql)
    print("Database schema applied successfully.")
