"""
Neo4j Knowledge Graph connection management using the official neo4j async driver.
Handles driver lifecycle, health checks, and Cypher query execution.
"""
from typing import Any, List, Dict
from neo4j import AsyncGraphDatabase, AsyncDriver
from agent.config import settings

# Global reference to the Neo4j async driver
_driver: AsyncDriver | None = None


async def get_graph_driver() -> AsyncDriver:
    """
    Initializes and returns the shared Neo4j AsyncDriver.
    Uses the Singleton pattern to avoid re-authenticating on every query.
    """
    global _driver
    if _driver is None:
        user = settings.neo4j_user
        # Support fallback if password was loaded under neo4j_password
        auth = (user, settings.neo4j_password)
        
        _driver = AsyncGraphDatabase.driver(
            uri=settings.neo4j_uri,
            auth=auth,
            max_connection_lifetime=3600,  # Refresh connections hourly
            max_connection_pool_size=10    # Match our Postgres pool concurrency
        )
    return _driver


async def close_graph_driver() -> None:
    """Closes the Neo4j driver on application shutdown."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None

async def check_graph_health() -> bool:
    """
    Verifies that Neo4j Aura is reachable and credentials are valid.
    Executes a lightweight Cypher statement: RETURN 1 AS health_check;
    """
    try:
        driver = await get_graph_driver()
        # Verify connectivity by opening an async session
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS health_check;")
            record = await result.single()
            return record["health_check"] == 1
    except Exception as e:
        print(f"Neo4j health check failed: {e}")
        return False


async def run_cypher(query: str, parameters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    Executes a Cypher query with parameters against Neo4j and returns results as Python dictionaries.
    
    Example:
        records = await run_cypher("MATCH (c:Company {name: $name}) RETURN c", {"name": "Google"})
    """
    driver = await get_graph_driver()
    async with driver.session() as session:
        result = await session.run(query, parameters or {})
        records = await result.data()
        return records
