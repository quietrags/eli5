ELI5 DSPy Agent
This project implements a simple but powerful "Explain Like I'm Five" (ELI5) agent using the DSPy framework. The agent can take any topic, fetch a summary from Wikipedia, and generate a simplified explanation suitable for a young child.

The agent demonstrates several key concepts in building modern AI systems:

Tool Use: The agent uses external Python scripts to fetch content and perform text processing.
LLM Reasoning: It uses dspy.ChainOfThought to encourage the language model to "think" before generating an answer.
Self-Correction: The agent uses a feedback loop to check the readability of its own output and iteratively refines it until it meets a target simplicity score.
Features
Dynamic Content: Explain any topic available on Wikipedia.
Text Preprocessing: Cleans fetched text by removing citations and other artifacts.
Intelligent Simplification: Uses DSPy's ChainOfThought for high-quality explanations.
Iterative Refinement: Automatically checks readability and simplifies further if needed.
Modular Tools: Built with separate, testable scripts for readability, content fetching, and preprocessing.
Project Structure
.
├── eli5_agent.py           # The main agent logic and DSPy module.
├── readability_checker.py  # Tool to calculate text readability scores.
├── wikipedia_fetcher.py    # Tool to fetch summaries from Wikipedia.
├── text_preprocessor.py    # Tool to clean and preprocess text.
├── .env                    # For storing API keys (e.g., OPENAI_API_KEY).
├── pyproject.toml          # Python dependencies managed by uv.
└── README.md               # This file.
Setup
Clone the repository:
bash
git clone <repository-url>
cd dspy-agent
Set up Python environment: This project uses uv for package management.
bash
# Initialize uv and create a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install dspy-ai python-dotenv wikipedia-api textstat
Configure API Key: Create a file named 
.env
 in the root of the project directory and add your OpenAI API key:
OPENAI_API_KEY="your-api-key-here"
Usage
You can run the agent from the command line. Provide a topic you want explained as an argument.

Syntax:

bash
python eli5_agent.py "Your Topic Here"
Example:

bash
# Activate the virtual environment first
source .venv/bin/activate

# Run the agent
python eli5_agent.py "Black Holes"
The agent will then:

Fetch the Wikipedia summary for "Black Holes".
Preprocess the text.
Generate an initial simplified explanation.
Check the readability. If it's too complex, it will refine it.
Print the final, easy-to-understand explanation and its readability scores.
How It Works
The agent's logic is orchestrated in 
eli5_agent.py
:

Fetch: It calls 
wikipedia_fetcher.py
 to get the summary of the requested topic.
Preprocess: It pipes the fetched text through 
text_preprocessor.py
 to remove unwanted brackets and extra spaces.
Simplify (Initial Pass): It uses a dspy.ChainOfThought module with a signature designed to produce an ELI5 explanation.
Evaluate & Refine: It calls 
readability_checker.py
 to get the Flesch-Kincaid grade level. If the level is above the target (default is 7.0), it enters a refinement loop.
Refine Loop: A second dspy.ChainOfThought module, with a signature specifically for making text even simpler, is used to refine the explanation. This loop continues until the target grade level is met or the maximum number of iterations is reached.