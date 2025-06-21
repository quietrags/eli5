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
    """You are an agent that explains complex topics to a five-year-old (ELI5).

    Your task is to take a topic, find information about it, and then explain it very simply, like you would to a five-year-old, using a structured format.

    **Your Reasoning Strategy**:
    1.  **Fetch Original Content**: Get the Wikipedia summary for the `topic`. (Tool: `fetch_wikipedia_summary`)
    2.  **Preprocess Original Content**: Clean the fetched text. (Tool: `preprocess_text`)
    3.  **Create Core Explanation**:
        a. Assess the readability of the preprocessed text. (Tool: `get_readability_scores`)
        b. **Iterative Simplification Loop**: If the grade level is too high (target: grade 4 or lower), you must simplify the text. 
           i. Identify a single, core complex jargon term from the *current version of the explanation text* that a five-year-old would not understand.
           ii. Use `define_jargon_term` to get a simple definition for that specific term.
           iii. Rewrite the *current version of the explanation text* to be simpler, incorporating the new definition and using very simple words. (This is your internal thought process to create the simplified text for the explanation section).
           iv. Re-assess the readability of your *newly rewritten explanation text* using `get_readability_scores`.
           v. Repeat this loop (steps 3.b.i - 3.b.iv) until the grade level of the explanation text is appropriate (grade 4 or lower) OR you determine that further jargon definition won't significantly improve its simplicity for a five-year-old. This simplified text will be the main explanation.
    4.  **Identify and Define Key Words**:
        a. From the *original preprocessed Wikipedia text*, identify 1 or 2 absolutely essential key terms that a five-year-old needs to know to understand the topic, even if they were already handled in the core explanation. These might be different from terms defined during iterative simplification if those were intermediate steps.
        b. For each identified key term, use `define_jargon_term` to get a simple definition. (Tool: `define_jargon_term`)
    5.  **Identify & Define Key Words**: After the explanation is simplified (grade level <= 4.0), identify 1-2 essential keywords from the simplified text that a five-year-old might not know. Use the `define_jargon_term` tool to get a simple definition for each of these keywords.
    6.  **Generate Factual Example**: After defining the keywords, generate a simple, factual, real-world example of the `topic`. Use the `generate_simple_example` tool.
    7.  **Assemble Final Output**: Combine the simplified core explanation, the keyword definitions, and the factual example into a single markdown formatted string for the `eli5_explanation` field. Also, create a `simplification_history` string that lists each simplification attempt and its readability score.

        **Final Explanation Format (`eli5_explanation`):**
        ```markdown
        **What is [Topic]?**
        [Your final simplified core explanation here.]

        **Key Words**
        **[Term 1]:** [Simple definition of Term 1]
        **[Term 2 (if any)]:** [Simple definition of Term 2]

        **For Example**
        [Your generated factual example here.]
        ```

        **History Format (`simplification_history`):**
        ```markdown
        **Simplification History**
        *   **Attempt 1 (Grade: [Score]):** "[Full text of first attempt]"
        *   **Attempt 2 (Grade: [Score]):** "[Full text of second attempt]"
        ...and so on.
        ```
    """
    topic = dspy.InputField(desc="The complex topic to be explained.")
    simplification_history = dspy.OutputField(desc="A log of each simplification attempt and its readability score.")
    eli5_explanation = dspy.OutputField(desc="The final, simplified explanation in markdown format.")


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
    # Load environment variables from .env file
    load_dotenv()

    # Set up the language model
    lm_model = dspy.LM(model="gpt-4.1", max_tokens=4000, temperature=0.7)
    dspy.settings.configure(lm=lm_model)

    # Set up the ReAct agent
    react_agent = dspy.ReAct(ELI5AgentSignature, tools=get_tools())
    console = Console()

    console.print("\n[bold cyan]ELI5 Agent Initialized.[/bold cyan]")
    console.print("Enter a topic you want explained, or type 'exit' to quit.")

    while True:
        topic = Prompt.ask("\n[bold]Enter a topic[/bold]")

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

            # You can inspect the full history of thoughts and tool use:
            # console.print("\n--- Agent's Thought Process ---")
            # lm_model.inspect_history(n=1)

        except Exception as e:
            console.print(f"[bold red]An error occurred while running the agent: {e}[/bold red]")

    print("\n--- Agent execution finished ---")


if __name__ == "__main__":
    main()
