import os
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from .base_llm import BaseLLM

# Load environment variables
load_dotenv()

class AnthropicLLM(BaseLLM):
    """
    Anthropic Claude LLM implementation.
    This class provides methods to interact with Anthropic's Claude API.
    """
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None, **kwargs):
        """
        Initialize Anthropic LLM instance.
        
        Args:
            model_name: Name of the Anthropic model to use
            api_key: Anthropic API key (if None, will try to load from environment)
            **kwargs: Additional model-specific parameters
        """
        super().__init__(model_name, **kwargs)
        
        # Use provided API key or load from environment
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set")
        
        # API endpoint
        self.api_endpoint = "https://api.anthropic.com/v1/messages"
        
        # Set the default max tokens if not provided
        self.max_tokens = kwargs.get("max_tokens", 4096)
    
    def call(self, prompt: str, temperature: float = 0.2, **kwargs) -> str:
        """
        Call Anthropic API with a prompt and return the response.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Controls randomness in the output (0-1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            The LLM's response as a string
        """
        # Prepare request headers
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Get max_tokens from kwargs or use default
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Prepare request data
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add any additional parameters from kwargs
        for key, value in kwargs.items():
            if key not in data and key != "max_tokens":
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
        response_data = response.json()
        return response_data["content"][0]["text"]
    
    def call_with_messages(self, messages: List[Dict[str, str]], temperature: float = 0.2, **kwargs) -> str:
        """
        Call Anthropic API with a list of messages and return the response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness in the output (0-1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            The LLM's response as a string
        """
        # Prepare request headers
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Get max_tokens from kwargs or use default
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Prepare request data
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add any additional parameters from kwargs
        for key, value in kwargs.items():
            if key not in data and key != "max_tokens":
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
        response_data = response.json()
        return response_data["content"][0]["text"] 