import numpy as np
from typing import Any 
import os
import requests
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

_ = load_dotenv()

class EmbeddingGenerator:
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize embedding generator with specified model.
        
        Args:
            model: Model name for embedding generation
        """
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

    def generate_embedding(self, text: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Generate embedding for a text using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": text
        }
        
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        embedding = response.json()["data"][0]["embedding"]
        return np.array(embedding, dtype=np.float64)
    
    def generate_features_embedding(self, features: dict[str, Any]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Generate embedding for problem features.
        
        Args:
            features: Dictionary containing problem features
            
        Returns:
            Embedding vector as numpy array
        """
        # Combine all feature texts into a single text for embedding
        combined_text = ""
        
        if "problem_type" in features:
            combined_text += f"Problem type: {features['problem_type']}\n"
        
        if "problem_description" in features:
            combined_text += f"Problem description: {features['problem_description']}\n"
        
        if "solution_steps" in features and isinstance(features["solution_steps"], list):
            combined_text += "Solution steps:\n"
            for i, step in enumerate(features["solution_steps"]):
                combined_text += f"Step {i+1}: {step}\n"
        
        if "used_techniques" in features and isinstance(features["used_techniques"], list):
            combined_text += "Techniques used:\n"
            for technique in features["used_techniques"]:
                combined_text += f"- {technique}\n"
        
        if "theorems_used" in features and isinstance(features["theorems_used"], list):
            combined_text += "Theorems used:\n"
            for theorem in features["theorems_used"]:
                combined_text += f"- {theorem}\n"
                
        # Generate embedding for the combined text
        return self.generate_embedding(combined_text) 