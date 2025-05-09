import os
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from .base_llm import BaseLLM

# Load environment variables
load_dotenv()

class OpenRouterLLM(BaseLLM):
    """
    OpenRouter LLM implementation.
    This class provides methods to interact with OpenRouter API,
    which allows access to multiple LLM providers through a unified API.
    """
    
    def __init__(
        self, 
        model_name: str = "deepseek/deepseek-prover-v2:free", 
        api_key: str="", **kwargs):
        """
        Initialize OpenRouter LLM instance.
        
        Args:
            model_name: Name of the model to use (format: "provider/model")
            api_key: OpenRouter API key (if None, will try to load from environment)
            **kwargs: Additional model-specific parameters
        """
        super().__init__(model_name, **kwargs)
        
        # Use provided API key or load from environment
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set")
        
        # API endpoint
        self.api_endpoint = "https://openrouter.ai/api/v1/chat/completions"
        
        # Set default HTTP referer if not provided
        # self.http_referer = kwargs.get("http_referer", "https://github.com/mathlearner")
    
    def call(self, prompt: str, temperature: float = 0.2, **kwargs) -> str:
        """
        Call OpenRouter API with a prompt and return the response.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Controls randomness in the output (0-1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            The LLM's response as a string
        """
        # Create message list with single user message
        messages = [{"role": "user", "content": prompt}]
        
        return self.call_with_messages(messages, temperature, **kwargs)
    
    def call_with_messages(self, messages: List[Dict[str, str]], temperature: float = 0.2, **kwargs) -> str:
        """
        Call OpenRouter API with a list of messages and return the response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness in the output (0-1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            The LLM's response as a string
        """
        # Prepare request headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # "HTTP-Referer": kwargs.get("http_referer", self.http_referer),
            # "X-Title": kwargs.get("x_title", "MathLearner")
        }
        
        # Prepare request data
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }
        
        # Add any additional parameters from kwargs
        for key, value in kwargs.items():
            if key not in data and key not in ["http_referer", "x_title"]:
                data[key] = value
        
        # Make the API request
        response = requests.post(
            self.api_endpoint,
            headers=headers,
            json=data
        )
        
        # Check for successful response
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        # Extract and return the content
        return response.json()["choices"][0]["message"]["content"]
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from OpenRouter.
        
        Returns:
            List of model information dictionaries
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            # "HTTP-Referer": self.http_referer
        }
        
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch models: {response.text}")
        
        return response.json()["data"] 
    
if __name__ == "__main__":
    llm = OpenRouterLLM()
    print(llm.get_available_models())