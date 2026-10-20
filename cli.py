"""
Interactive Command Line Interface (CLI) for Agentic RAG.
Connects to the FastAPI backend and provides real-time streaming chat.
"""
import sys
import json
import asyncio
import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

async def stream_chat(message: str, api_url: str = "http://localhost:8058/chat/stream"):
    """
    Sends a message to the FastAPI SSE streaming endpoint and prints clean tokens.
    """
    payload = {"message": message}
    full_response = []
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", api_url, json=payload) as response:
                if response.status_code != 200:
                    console.print(f"[bold red]API Error ({response.status_code}):[/bold red] Failed to connect.")
                    return

                console.print("\n[bold cyan]Assistant:[/bold cyan] ", end="")
                
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if not data_str:
                            continue
                        try:
                            data = json.loads(data_str)
                            if "token" in data:
                                token = data["token"]
                                full_response.append(token)
                                console.print(token, end="")
                                sys.stdout.flush()
                            elif data.get("done"):
                                console.print("\n")
                                break
                            elif "error" in data:
                                console.print(f"\n[bold red]Error: {data['error']}[/bold red]\n")
                                break
                        except json.JSONDecodeError:
                            continue

    except httpx.ConnectError:
        console.print("\n[bold red]Could not connect to API server at http://localhost:8058.[/bold red]\n")
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error: {e}[/bold red]\n")


async def main():
    """Main interactive chat loop."""
    console.print(
        Panel.fit(
            "[bold green]Agentic RAG with Knowledge Graph[/bold green]\n"
            "[dim]Powered by Neon pgvector + Neo4j Aura + Groq Llama/GPT-OSS[/dim]\n\n"
            "Commands:\n"
            "  [bold cyan]exit[/bold cyan] or [bold cyan]quit[/bold cyan] : Exit the chat\n"
            "  [bold cyan]clear[/bold cyan]        : Clear terminal screen",
            title="Welcome!!!",
            border_style="green"
        )
    )

    while True:
        try:
            user_input = Prompt.ask("\n[bold yellow]You[/bold yellow]")
            user_input = user_input.strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "clear":
                console.clear()
                continue

            await stream_chat(user_input)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session closed.[/dim]")
            break


if __name__ == "__main__":
    asyncio.run(main())
