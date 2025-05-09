import os
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from .base_llm import BaseLLM

# Load environment variables
load_dotenv()

class OpenAILLM(BaseLLM):
    """
    OpenAI LLM implementation.
    This class provides methods to interact with OpenAI API.
    """
    
    def __init__(self, model_name: str = "qwen/qwen3-32b:free", api_key: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI LLM instance.
        
        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (if None, will try to load from environment)
            **kwargs: Additional model-specific parameters
        """
        super().__init__(model_name, **kwargs)
        
        # Use provided API key or load from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        
        # API endpoint
        self.api_endpoint = "https://api.openai.com/v1/chat/completions"
    
    def call(self, prompt: str, temperature: float = 0.2, **kwargs) -> str:
        """
        Call OpenAI API with a prompt and return the response.
        
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
        Call OpenAI API with a list of messages and return the response.
        
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
            "Content-Type": "application/json"
        }
        
        # Prepare request data
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }
        
        # Add any additional parameters from kwargs
        data.update({k: v for k, v in kwargs.items() if k not in data})
        
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