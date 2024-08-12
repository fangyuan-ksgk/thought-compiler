from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from typing import Callable, Dict

# Import the original build_research_graph function
from src.utils import build_research_graph

console = Console()

def rich_get_action() -> str:
    return Prompt.ask("Do you like this paper?", choices=["y", "n", "back"], default="y")

def rich_get_reject_feedback() -> str:
    feedback = Prompt.ask("Please provide feedback for rejection")
    console.print(f"Feedback received: {feedback}", style="italic yellow")
    return feedback

def rich_print_paper(paper: Dict):
    title = Text(f"Title: {paper['title']}", style="bold")
    score = Text(f"Score: {paper['score']}", style="cyan")
    comment = Text(f"AI Comment: {paper['comment']}", style="green")
    panel = Panel(f"{title}\n{score}\n{comment}", border_style="blue")
    console.print(panel)

def rich_print_action_error():
    console.print("Invalid action. Please choose y, n, or back.", style="bold red")

def rich_print_summary_begin():
    console.print(Panel("Failed to recommend papers, reflecting on human preference", style="bold yellow"))

def rich_print_summary(summary: str):
    console.print(Panel(f"Summary of human preference:\n{summary}", style="italic"))

def main():
    console.print(Panel("LangGraph Research Assistant", style="bold magenta"))

    # Call the original build_research_graph function with Rich console inputs
    graph = build_research_graph(
        get_action=rich_get_action,
        get_reject_feedback=rich_get_reject_feedback,
        print_paper=rich_print_paper,
        print_action_error=rich_print_action_error,
        print_summary_begin=rich_print_summary_begin,
        print_summary=rich_print_summary
    )

    # Initial state (you may need to adjust this based on your actual implementation)
    state = {
        "papers": [],
        "messages": []
    }

    # Run the graph
    # rich_get_reject_feedback() # this works alright
    
    for output in graph.stream(state):
        # Process output as needed
        pass

    console.print(Panel("Research complete!", style="bold green"))

if __name__ == "__main__":
    main()