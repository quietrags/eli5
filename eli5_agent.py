import dspy
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

# Import the standalone tool functions from our refactored tools.py file
from tools import (
    define_jargon_term,
    fetch_wikipedia_summary,
    fetch_simple_wikipedia_summary,
    get_readability_scores,
    preprocess_text,
    generate_simple_example
)

class ELI5AgentSignature(dspy.Signature):
    """You are an expert at explaining complex topics to five-year-olds.

    Your goal: Take any topic and create a simple, engaging explanation that a kindergartner can understand.

    Use the available tools to:
    - Research the topic thoroughly
    - Simplify complex language and concepts  
    - Check that your explanation is at an appropriate reading level (grade 4 or lower)
    - Define important words a child might not know
    - Provide real examples they can relate to

    Format your final explanation as:
    
    **What is [Topic]?**
    [Simple explanation in 2-3 sentences]

    **Key Words**
    **[Word]:** [Simple definition]
    
    **For Example**
    [Real-world example]

    Keep iterating until the explanation is truly simple enough for a 5-year-old!
    """
    topic = dspy.InputField(desc="The complex topic to explain simply.")
    simplification_history = dspy.OutputField(desc="Log of simplification attempts with readability scores.")
    eli5_explanation = dspy.OutputField(desc="Final child-friendly explanation in markdown format.")


def get_tools():
    return [
        # List of tools for the ReAct agent
        dspy.Tool(fetch_wikipedia_summary),
        dspy.Tool(fetch_simple_wikipedia_summary),
        dspy.Tool(get_readability_scores),
        dspy.Tool(preprocess_text),
        dspy.Tool(define_jargon_term),
        dspy.Tool(generate_simple_example) # Added Example Generator tool
    ]


def main():
    import sys
    
    # Load environment variables from .env file
    load_dotenv()

    # Set up the language model
    lm_model = dspy.LM(model="gpt-4.1", max_tokens=4000, temperature=0.7)
    dspy.settings.configure(lm=lm_model)

    # Set up the ReAct agent
    react_agent = dspy.ReAct(ELI5AgentSignature, tools=get_tools())
    console = Console()

    # Check if topic provided as command line argument
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
        console.print(f"\n[bold cyan]Testing ELI5 Agent with topic: '{topic}'[/bold cyan]")
        
        try:
            response = react_agent(topic=topic)

            # Print the simplification history first, rendered as markdown
            console.print("\n" + "-"*50)
            console.print("[bold]Simplification History:[/bold]")
            history_md = Markdown(response.simplification_history, style="")
            console.print(history_md)
            
            # Print the final explanation using rich for beautiful markdown rendering
            console.print("\n" + "-"*50)
            console.print("[bold]Final Explanation:[/bold]")
            md = Markdown(response.eli5_explanation, style="")
            console.print(md)
            console.print("-"*50 + "\n")

        except Exception as e:
            console.print(f"[bold red]An error occurred while running the agent: {e}[/bold red]")
        
        return

    # Interactive mode
    console.print("\n[bold cyan]ELI5 Agent Initialized.[/bold cyan]")
    console.print("Enter a topic you want explained, or type 'exit' to quit.")

    while True:
        try:
            topic = Prompt.ask("\n[bold]Enter a topic[/bold]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold cyan]Goodbye![/bold cyan]")
            break

        if topic.lower() in ["exit", "quit"]:
            console.print("[bold cyan]Goodbye![/bold cyan]")
            break

        console.print(f"\n[bold]Running ReAct Agent for topic: '{topic}'...[/bold]")
        try:
            response = react_agent(topic=topic)

            # Print the simplification history first, rendered as markdown
            console.print("\n" + "-"*50)
            console.print("[bold]Simplification History:[/bold]")
            history_md = Markdown(response.simplification_history, style="")
            console.print(history_md)
            
            # Print the final explanation using rich for beautiful markdown rendering
            console.print("\n" + "-"*50)
            console.print("[bold]Final Explanation:[/bold]")
            md = Markdown(response.eli5_explanation, style="")
            console.print(md)
            console.print("-"*50 + "\n")

        except Exception as e:
            console.print(f"[bold red]An error occurred while running the agent: {e}[/bold red]")

    print("\n--- Agent execution finished ---")


if __name__ == "__main__":
    main()
