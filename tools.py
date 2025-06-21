import dspy
import subprocess
import re

# --- Jargon Definer Tool --- #

# Signature for the Jargon Definer LLM call (internal to the tool function)
class _DefineJargonSignature(dspy.Signature):
    """Explain the jargon term in one simple sentence, like you're talking to a five-year-old."""
    term: str = dspy.InputField(desc="The jargon term to be defined.")
    simple_definition: str = dspy.OutputField(desc="A simple, one-sentence definition.")

def define_jargon_term(term: str) -> str:
    """Takes a complex word or term and returns a simple, one-sentence definition suitable for a five-year-old.
    Args:
        term (str): The jargon term to be defined.
    Returns:
        str: A simple, one-sentence definition of the term.
    """
    try:
        if not dspy.settings.lm:
            return "Error: DSPy Language Model not configured. Cannot define jargon."
        predictor = dspy.Predict(_DefineJargonSignature)
        response = predictor(term=term)
        return str(response.simple_definition)
    except Exception as e:
        return f"Error defining term '{term}': {str(e)}"

# --- Wikipedia Fetcher Tool --- #

def fetch_wikipedia_summary(topic: str) -> str:
    """Fetches the introduction summary of a Wikipedia article for a given topic. Limits summary to approx 1500 chars.
    Args:
        topic (str): The topic to search for on Wikipedia.
    Returns:
        str: The summary of the Wikipedia article, or an error message if not found or an issue occurs.
    """
    try:
        py_command = (
            f"import wikipediaapi; "
            f"wiki = wikipediaapi.Wikipedia('MyEli5Agent/1.0', 'en'); "
            f"page = wiki.page('{topic}'); "
            f"print(page.summary[:1500] if page.exists() else 'Topic not found on Wikipedia.')"
        )
        result = subprocess.run(
            ["python", "-c", py_command],
            capture_output=True, text=True, check=True, timeout=20
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error fetching from Wikipedia for '{topic}': {str(e)}"

# --- Simple Wikipedia Fetcher Tool --- #

def fetch_simple_wikipedia_summary(topic: str) -> str:
    """Fetches the introduction summary of a Simple English Wikipedia article for a given topic. Limits summary to approx 1500 chars.
    Args:
        topic (str): The topic to search for on Simple English Wikipedia.
    Returns:
        str: The summary of the Simple English Wikipedia article, or an error message.
    """
    try:
        py_command = (
            f"import wikipediaapi; "
            f"wiki = wikipediaapi.Wikipedia('MyEli5Agent/1.0', 'simple'); "
            f"page = wiki.page('{topic}'); "
            f"print(page.summary[:1500] if page.exists() else 'Topic not found on Simple Wikipedia.')"
        )
        result = subprocess.run(
            ["python", "-c", py_command],
            capture_output=True, text=True, check=True, timeout=20
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error fetching from Simple Wikipedia for '{topic}': {str(e)}"

# --- Readability Checker Tool --- #

def get_readability_scores(text_to_check: str) -> str:
    """Calculates readability scores for a given text, primarily the Flesch-Kincaid grade level.
    Args:
        text_to_check (str): The text whose readability needs to be checked.
    Returns:
        str: A summary of the readability score, e.g., 'Flesch-Kincaid Grade Level: 8.5'.
    """
    try:
        py_command = "import sys; import textstat; text = sys.stdin.read(); print(f'Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(text)}')"
        result = subprocess.run(
            ["python", "-c", py_command],
            input=text_to_check,
            capture_output=True, text=True, check=True, timeout=10
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error checking readability: {str(e)}"

# --- Text Preprocessor Tool --- #

def preprocess_text(raw_text: str) -> str:
    """Cleans text by removing common Wikipedia artifacts like [1], [edit], (listen) annotations, and normalizes whitespace.
    Args:
        raw_text (str): The raw text to be preprocessed.
    Returns:
        str: The text after removing specified artifacts and normalizing whitespace.
    """
    try:
        text = raw_text
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*\(listen\)\s*', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*\([^)]*\b(?:e\.g\.|i\.e\.|etc\.|cit\.(?:\s*needed)?|citation needed|[\w\s]*\d+:\d+)[^)]*\)\s*', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\(\s*\)', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        return f"Error during preprocessing: {str(e)}"

# --- Example Generator Tool --- #

class FactualExampleGeneratorSignature(dspy.Signature):
    """ATTENTION: Your task is to provide 1-2 SIMPLE, FACTUAL, REAL-WORLD examples of the given 'topic'. DO NOT PROVIDE ANALOGIES or metaphorical comparisons. Only list actual instances or types.
    The examples must be suitable for a five-year-old: very short, easy to understand, and concrete.

    Example 1:
    Topic: Volcanoes
    Factual Examples Output: "Mount St. Helens is a volcano in America that erupted. Kilauea in Hawaii is another volcano that has lava flowing often."

    Example 2:
    Topic: Planets
    Factual Examples Output: "We live on a planet called Earth. Mars is a red planet that scientists send robots to."

    Example 3:
    Topic: Dinosaurs
    Factual Examples Output: "T-Rex was a giant dinosaur with big teeth that ate meat. Stegosaurus was a dinosaur with plates on its back."
    """
    concept = dspy.InputField(desc="(Optional) A brief, simplified core idea related to the topic, for context. The main focus for example generation should be the 'topic' itself.")
    topic = dspy.InputField(desc="The specific subject for which factual examples are needed (e.g., 'Volcanoes', 'Rivers', 'Mammals').")
    factual_instances_of_topic = dspy.OutputField(desc="Provide specific, real-world, named examples of the 'topic'. DO NOT use analogies, metaphors, or comparisons. For example, if the topic is 'Volcanoes', a correct output is 'Mount Fuji in Japan and Mount Vesuvius in Italy'. An incorrect output would be 'A volcano is like a shaken soda bottle'.")

def generate_simple_example(concept: str, topic: str) -> str:
    """Generates a simple, concrete, and factual real-world example for a given topic.
    This tool is designed to be used within a DSPy ReAct agent to provide grounding for complex topics.
    It explicitly avoids analogies and metaphors in favor of named, verifiable examples.

    Args:
        concept: A brief, simplified core idea related to the topic. This is used for context but the primary focus is the 'topic'.
        topic: The specific subject for which factual examples are needed (e.g., 'Volcanoes', 'Rivers', 'Mammals').

    Returns:
        A string containing one or two simple, concrete, factual examples of the topic.
    """
    try:
        # Ensure DSPy is configured
        if not dspy.settings.lm:
            # Fallback or minimal configuration if not already set
            # This is a safeguard; ideally, configuration is done globally once.
            print("Warning: DSPy LM not configured. Attempting fallback configuration for ExampleGeneratorTool.")
            # Attempt to configure with a default or previously set model if possible
            # This part might need adjustment based on how global config is handled
            # For now, assuming it might have been configured elsewhere or raising an error
            # if no lm is available, dspy.Predict will fail anyway.

        predictor = dspy.Predict(FactualExampleGeneratorSignature)
        # Handle potential API errors or unexpected LLM responses
        result = predictor(concept=concept, topic=topic)
        if hasattr(result, 'factual_instances_of_topic') and result.factual_instances_of_topic:
            return result.factual_instances_of_topic
        else:
            # Fallback if the LLM doesn't produce a valid example
            print(f"Warning: ExampleGeneratorTool for concept '{concept}' and topic '{topic}' did not receive a valid example from LLM. Returning a placeholder.")
            return f"Imagine you are learning about {topic}. One cool thing is {concept}. For instance, think about how that works in everyday life!"

    except Exception as e:
        print(f"Error in ExampleGeneratorTool for concept '{concept}', topic '{topic}': {e}")
        # Provide a generic fallback example in case of any error
        return f"Let's think about {topic}. We know that {concept}. For example, you can see this when... (oops, I need to think of a better example!)"

# --- Analogy Generator Tool --- #

class _GenerateAnalogySignature(dspy.Signature):
    """Generate a simple, one-sentence analogy for a complex concept to help a five-year-old understand it."""
    concept = dspy.InputField(desc="The complex concept to explain.")
    analogy = dspy.OutputField(desc="A simple, relatable analogy.")

def generate_analogy(concept: str) -> str:
    """Generates a simple, child-friendly analogy for a complex concept.
    Args:
        concept (str): The concept to create an analogy for.
    Returns:
        str: A simple, one-sentence analogy.
    """
    try:
        if not dspy.settings.lm:
            return "Error: DSPy Language Model not configured. Cannot generate analogy."
        predictor = dspy.Predict(_GenerateAnalogySignature)
        response = predictor(concept=concept)
        return str(response.analogy)
    except Exception as e:
        return f"Error generating analogy for '{concept}': {str(e)}"
