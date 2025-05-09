from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union


class BaseLLM(ABC):
    """
    Abstract base class for LLM connections.
    This class defines the interface for interacting with different LLM providers.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM instance.
        
        Args:
            model_name: Name of the LLM model to use
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.model_params = kwargs
    
    @abstractmethod
    def call(self, prompt: str, temperature: float = 0.2, **kwargs) -> str:
        """
        Call the LLM with a prompt and return the response.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Controls randomness in the output (0-1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            The LLM's response as a string
        """
        pass
    
    @abstractmethod
    def call_with_messages(self, messages: List[Dict[str, str]], temperature: float = 0.2, **kwargs) -> str:
        """
        Call the LLM with a list of messages and return the response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness in the output (0-1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            The LLM's response as a string
        """
        pass
    
    def create_prompt_from_template(self, template: str, **kwargs) -> str:
        """
        Create a prompt from a template and variables.
        
        Args:
            template: The prompt template with placeholders
            **kwargs: Variables to insert into the template
            
        Returns:
            The formatted prompt
        """
        return template.format(**kwargs)

    def __str__(self) -> str:
        """String representation of the LLM instance."""
        return f"{self.__class__.__name__}(model_name={self.model_name})" 