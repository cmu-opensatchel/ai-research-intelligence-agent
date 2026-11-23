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
    use_llm_verification: bool = typer.Option(False, help="Enable Qwen3-8B local LLM for claim verification"),
):
    """Run the AI research intelligence workflow."""
    topic_list = [t.strip() for t in topics.split(",") if t.strip()]
    url_list = [u.strip() for u in urls.split(",") if u.strip()]

    console.print(Panel.fit(f"[bold]AI Research Intelligence Agent[/bold]\nTopics: {', '.join(topic_list)}\nWindow: {window} days\nSocial discovery: {'enabled' if social else 'disabled'}\nLLM verification: {'enabled' if use_llm_verification else 'disabled'}"))

    cfg = AgentConfig()
    cfg.enable_social_discovery = social
    cfg.use_llm_verification = use_llm_verification

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
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Title", style="white", width=50)
        table.add_column("Source", style="magenta", width=30)
        table.add_column("Link", style="cyan", width=40)
        table.add_column("Assessment", style="green")
        table.add_column("Score", style="yellow")

        # Build a lookup from candidate_id to candidate for readable titles
        candidate_lookup = {c.id: c for c in result["candidates"]}

        for i, entry in enumerate(result["rankings"], 1):
            cand = candidate_lookup.get(entry["candidate_id"])
            title = cand.title if cand else entry.get("candidate_id", "")[:8]
            source_title = cand.source.title if cand and cand.source else ""
            source_url = cand.source.url if cand and cand.source else ""
            # Truncate long display strings
            title_disp = (title[:47] + "…") if len(title) > 47 else title
            source_disp = (source_title[:27] + "…") if len(source_title) > 27 else source_title
            url_disp = (source_url[:37] + "…") if len(source_url) > 37 else source_url
            table.add_row(
                str(i), title_disp, source_disp, url_disp,
                entry["best_assessment"], str(entry["best_score"]),
            )
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