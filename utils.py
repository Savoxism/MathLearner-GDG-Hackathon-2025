import os
import json
import numpy as np
import pickle
import requests
from dotenv import load_dotenv
from typing import Any, List, Dict, Tuple, Optional

from llm import LLMFactory

load_dotenv()

def load_math_dataset(file_path: str) -> list[dict[str, Any]]:
    """Load the MATH dataset from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def call_llm_api(prompt: str, model: str = "qwen/qwen3-32b:free") -> str:
    """
    Call LLM API with a prompt and return the response.
    This is a legacy function that uses the new LLM classes internally.
    
    Args:
        prompt: The prompt to send to the LLM
        model: Model name to use
        
    Returns:
        The LLM's response as a string
    """
    # Determine provider based on model name
    provider = "openai"  # Default provider
    
    if model.startswith("claude"):
        provider = "anthropic"
    elif "/" in model:
        # OpenRouter models typically have format "provider/model"
        provider = "openrouter"
    
    # Create LLM instance using factory
    llm = LLMFactory.create(provider, model_name=model)
    
    # Call the LLM and return the response
    return llm.call(prompt)

def read_prompt_template(template_name: str) -> str:
    """
    Read a prompt template from a file.
    
    Args:
        template_name: Name of the template file (without .txt extension)
        
    Returns:
        The template content as a string
    """
    template_path = os.path.join("prompt", f"{template_name}.txt")
    
    if not os.path.exists(template_path):
        raise ValueError(f"Template file not found: {template_path}")
    
    with open(template_path, 'r') as f:
        template = f.read()
    
    return template

def extract_python_code(text: str) -> str:
    """Extract Python code from LLM response."""
    if "```python" in text:
        # Extract code between ```python and ```
        start_idx = text.find("```python") + len("```python")
        end_idx = text.find("```", start_idx)
        return text[start_idx:end_idx].strip()
    elif "```" in text:
        # Extract code between ``` and ```
        start_idx = text.find("```") + len("```")
        end_idx = text.find("```", start_idx)
        return text[start_idx:end_idx].strip()
    else:
        # Return the whole text if no code block markers found
        return text

def execute_python_code(code: str, problem_input: Optional[str] = None) -> Tuple[bool, Any, str]:
    """
    Execute Python code and return success status, result, and error message.
    
    Args:
        code: Python code to execute
        problem_input: Optional input to provide to the Python code
        
    Returns:
        Tuple of (success, result, error_message)
    """
    try:
        local_vars = {}
        if problem_input:
            local_vars['problem_input'] = problem_input
        
        # Add timeout mechanism to prevent infinite loops
        exec(code, globals(), local_vars)
        
        if 'answer' in local_vars:
            return True, local_vars['answer'], ""
        elif 'result' in local_vars:
            return True, local_vars['result'], ""
        else:
            return True, None, "Code executed but no 'answer' or 'result' variable found."
    
    except Exception as e:
        return False, None, str(e)

def save_to_vector_db(problem_id: str, problem_features: Dict[str, Any], embeddings: np.ndarray[Any, np.dtype[np.float64]], db_path: str):
    """Save problem features and embeddings to a vector database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Simple file-based vector DB
    db = {}
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            db = pickle.load(f)
    
    db[problem_id] = {
        "features": problem_features,
        "embeddings": embeddings
    }
    
    with open(db_path, 'wb') as f:
        pickle.dump(db, f)

def cosine_similarity(a: np.ndarray[Any, np.dtype[np.float64]], b: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_similar_problems(query_embedding: np.ndarray[Any, np.dtype[np.float64]], db_path: str, k: int = 3) -> List[Dict[str, Any]]:
    """Find k most similar problems from the vector database."""
    if not os.path.exists(db_path):
        return []
    
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    
    similarities = []
    for problem_id, problem_data in db.items():
        embedding = problem_data["embeddings"]
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((problem_id, similarity, problem_data["features"]))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k
    return [{"id": pid, "similarity": sim, "features": feat} for pid, sim, feat in similarities[:k]] 