import asyncio
from agent.config import settings
from neo4j import AsyncGraphDatabase

async def main():
    print(f"Connecting to: {settings.neo4j_uri}")
    for username in [settings.neo4j_user, "neo4j", "21d928a9"]:
        print(f"Testing username: '{username}'...")
        try:
            driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(username, settings.neo4j_password)
            )
            async with driver.session() as session:
                result = await session.run("RETURN 'Connected successfully!' AS message")
                record = await result.single()
                print(f"🎉 SUCCESS! {record['message']} (Username that worked: {username})")
                await driver.close()
                return
        except Exception as e:
            print(f"❌ Failed with username '{username}': {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
