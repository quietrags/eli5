import dspy
import subprocess
import re
import os
import json
from pathlib import Path

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
    if not term or not term.strip():
        return "Error: No term provided for definition"
    
    try:
        if not dspy.settings.lm:
            return "Error: DSPy Language Model not configured. Cannot define jargon."
        
        predictor = dspy.Predict(_DefineJargonSignature)
        response = predictor(term=term.strip())
        
        if hasattr(response, 'simple_definition') and response.simple_definition:
            return str(response.simple_definition).strip()
        else:
            return f"Could not generate definition for '{term}'"
            
    except Exception as e:
        return f"Error defining term '{term}': {str(e)}"

# --- Cache Management --- #

def _get_cache_path():
    """Get the cache directory path."""
    cache_dir = Path.home() / ".eli5_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def _get_cached_content(cache_key: str) -> str:
    """Retrieve cached content if it exists."""
    cache_file = _get_cache_path() / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('content', '')
        except:
            pass
    return ""

def _cache_content(cache_key: str, content: str):
    """Cache content to disk."""
    cache_file = _get_cache_path() / f"{cache_key}.json"
    try:
        with open(cache_file, 'w') as f:
            json.dump({'content': content}, f)
    except:
        pass

def _safe_wikipedia_fetch(topic: str, wiki_type: str = 'simple', max_retries: int = 2) -> str:
    """Safely fetch Wikipedia content with retries and caching."""
    cache_key = f"{wiki_type}_{topic.lower().replace(' ', '_')}"
    
    # Check cache first
    cached = _get_cached_content(cache_key)
    if cached and not cached.startswith("Error"):
        return cached
    
    # Try fetching with retries
    for attempt in range(max_retries):
        try:
            py_command = (
                f"import wikipediaapi; "
                f"wiki = wikipediaapi.Wikipedia('MyEli5Agent/1.0', '{wiki_type}'); "
                f"page = wiki.page('{topic}'); "
                f"print(page.summary[:1500] if page.exists() else 'Topic not found on {wiki_type} Wikipedia.')"
            )
            result = subprocess.run(
                ["python", "-c", py_command],
                capture_output=True, text=True, check=True, timeout=25
            )
            content = result.stdout.strip()
            
            # Cache successful results
            if content and not content.startswith("Topic not found"):
                _cache_content(cache_key, content)
            
            return content
            
        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                continue
            return f"Error: Wikipedia fetch timed out for '{topic}' after {max_retries} attempts"
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            return f"Error fetching from {wiki_type} Wikipedia for '{topic}': {str(e)}"
    
    return f"Error: Failed to fetch '{topic}' after {max_retries} attempts"

# --- Intelligent Wikipedia Fetcher Tool --- #

def fetch_wikipedia_summary(topic: str) -> str:
    """Intelligently fetches Wikipedia content, trying Simple English first, then regular Wikipedia.
    Includes caching and retry logic for reliability.
    
    Args:
        topic (str): The topic to search for on Wikipedia.
    Returns:
        str: The best available Wikipedia summary, or an error message.
    """
    # Try Simple English Wikipedia first (better for ELI5)
    simple_result = _safe_wikipedia_fetch(topic, 'simple')
    
    # If Simple Wikipedia has content, use it
    if simple_result and not simple_result.startswith("Topic not found") and not simple_result.startswith("Error"):
        return f"[Simple Wikipedia] {simple_result}"
    
    # Fall back to regular Wikipedia
    regular_result = _safe_wikipedia_fetch(topic, 'en')
    
    if regular_result and not regular_result.startswith("Topic not found") and not regular_result.startswith("Error"):
        return f"[Wikipedia] {regular_result}"
    
    # If both failed, return the most informative error
    if "not found" in simple_result and "not found" in regular_result:
        return f"Topic '{topic}' not found on either Simple or regular Wikipedia. Try a different topic or check spelling."
    
    return f"Error accessing Wikipedia for '{topic}'. Please try again later."

# --- Legacy Simple Wikipedia Tool (kept for compatibility) --- #

def fetch_simple_wikipedia_summary(topic: str) -> str:
    """Fetches only Simple English Wikipedia. Use fetch_wikipedia_summary() for intelligent selection.
    Args:
        topic (str): The topic to search for on Simple English Wikipedia.
    Returns:
        str: The summary of the Simple English Wikipedia article, or an error message.
    """
    return _safe_wikipedia_fetch(topic, 'simple')

# --- Readability Checker Tool --- #

def get_readability_scores(text_to_check: str) -> str:
    """Calculates readability scores for a given text, primarily the Flesch-Kincaid grade level.
    Args:
        text_to_check (str): The text whose readability needs to be checked.
    Returns:
        str: A summary of the readability score, e.g., 'Flesch-Kincaid Grade Level: 8.5'.
    """
    if not text_to_check or not text_to_check.strip():
        return "Error: No text provided for readability check"
    
    try:
        py_command = "import sys; import textstat; text = sys.stdin.read(); print(f'Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(text)}')"
        result = subprocess.run(
            ["python", "-c", py_command],
            input=text_to_check,
            capture_output=True, text=True, check=True, timeout=15
        )
        output = result.stdout.strip()
        if not output:
            return "Error: No readability score calculated"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Readability check timed out"
    except subprocess.CalledProcessError as e:
        return f"Error calculating readability: {e.stderr.strip() if e.stderr else 'Unknown error'}"
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
    if not raw_text:
        return "Error: No text provided for preprocessing"
    
    try:
        text = str(raw_text)
        
        # Remove Wikipedia-specific artifacts
        text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers [1], [2], etc.
        text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)  # Remove [edit] links
        text = re.sub(r'\s*\(listen\)\s*', ' ', text, flags=re.IGNORECASE)  # Remove (listen) audio links
        
        # Remove citation needed and similar parenthetical notes
        text = re.sub(r'\s*\([^)]*\b(?:e\.g\.|i\.e\.|etc\.|cit\.(?:\s*needed)?|citation needed|[\w\s]*\d+:\d+)[^)]*\)\s*', ' ', text, flags=re.IGNORECASE)
        
        # Remove empty parentheses
        text = re.sub(r'\(\s*\)', ' ', text)
        
        # Remove source prefixes if present
        text = re.sub(r'^\[(?:Simple )?Wikipedia\]\s*', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return "Error: Text became empty after preprocessing"
            
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
    if not topic or not topic.strip():
        return "Error: No topic provided for example generation"
    
    try:
        if not dspy.settings.lm:
            return "Error: DSPy Language Model not configured. Cannot generate examples."

        predictor = dspy.Predict(FactualExampleGeneratorSignature)
        result = predictor(concept=concept or "", topic=topic.strip())
        
        if hasattr(result, 'factual_instances_of_topic') and result.factual_instances_of_topic:
            return str(result.factual_instances_of_topic).strip()
        else:
            return f"Could not generate specific examples for '{topic}'. Try asking about a more common topic."

    except Exception as e:
        return f"Error generating examples for '{topic}': {str(e)}"

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
    if not concept or not concept.strip():
        return "Error: No concept provided for analogy generation"
    
    try:
        if not dspy.settings.lm:
            return "Error: DSPy Language Model not configured. Cannot generate analogy."
        
        predictor = dspy.Predict(_GenerateAnalogySignature)
        response = predictor(concept=concept.strip())
        
        if hasattr(response, 'analogy') and response.analogy:
            return str(response.analogy).strip()
        else:
            return f"Could not generate analogy for '{concept}'"
            
    except Exception as e:
        return f"Error generating analogy for '{concept}': {str(e)}"
