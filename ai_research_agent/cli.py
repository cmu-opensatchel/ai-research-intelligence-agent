"""Command-line interface for the AI Research Intelligence Agent.

Usage:
    python -m ai_research_agent.cli run --topics "agentic AI,LLMs,RAG" --window 30
    python -m ai_research_agent.cli run --topics "GNNs" --window 7
    python -m ai_research_agent.cli history
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .coordinator import run_workflow
from .config import AgentConfig
from .storage import RunStore

app = typer.Typer(help="AI Research Intelligence Agent - CLI")
console = Console()


@app.command()
def run(
    topics: str = typer.Option("agentic AI,LLMs,RAG,multimodal AI,GNNs", help="Comma-separated AI topics"),
    window: int = typer.Option(30, help="Reporting window in days"),
    urls: str = typer.Option("", help="Comma-separated lab announcement URLs (optional)"),
    social: bool = typer.Option(False, help="Enable social media discovery (requires API keys; discovery signals only)"),
):
    """Run the AI research intelligence workflow."""
    topic_list = [t.strip() for t in topics.split(",") if t.strip()]
    url_list = [u.strip() for u in urls.split(",") if u.strip()]

    console.print(Panel.fit(f"[bold]AI Research Intelligence Agent[/bold]\nTopics: {', '.join(topic_list)}\nWindow: {window} days\nSocial discovery: {'enabled' if social else 'disabled'}"))

    cfg = AgentConfig()
    cfg.enable_social_discovery = social

    result = run_workflow(
        topics=topic_list,
        reporting_window_days=window,
        announcement_urls=url_list if url_list else None,
        config=cfg,
    )

    # Display roundup
    console.print("\n")
    console.print(Panel(result["roundup"], title="Roundup Report", border_style="green"))

    # Display rankings table
    if result["rankings"]:
        table = Table(title="Candidate Rankings")
        table.add_column("Rank", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Assessment", style="magenta")
        table.add_column("Score", style="yellow")
        for i, entry in enumerate(result["rankings"], 1):
            table.add_row(str(i), entry.get("candidate_id", "")[:8], entry["best_assessment"], str(entry["best_score"]))
        console.print(table)

    # Display errors if any
    if result["errors"]:
        console.print("\n[bold yellow]Errors/Warnings:[/bold yellow]")
        for e in result["errors"]:
            console.print(f"  - {e}")

    # Artifact info
    if result["artifact_path"]:
        console.print(f"\n[bold]Run artifact saved to:[/bold] {result['artifact_path']}")
    console.print(f"[bold]Termination reason:[/bold] {result['termination_reason']}")


@app.command()
def history():
    """Show run history."""
    store = RunStore()
    runs = store.list_runs()
    if not runs:
        console.print("No runs recorded yet.")
        return
    table = Table(title="Run History")
    table.add_column("Run ID", style="cyan")
    table.add_column("Timestamp", style="white")
    table.add_column("Topics", style="magenta")
    table.add_column("Window (days)", style="yellow")
    for r in runs:
        table.add_row(r["run_id"][:8], r["timestamp"], r["topics"][:50], str(r["reporting_window_days"]))
    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()