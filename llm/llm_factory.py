from typing import Optional, Dict, Any, List

from .base_llm import BaseLLM
from .openai_llm import OpenAILLM
from .anthropic_llm import AnthropicLLM
from .openrouter_llm import OpenRouterLLM


class LLMFactory:
    """
    Factory class for creating LLM instances.
    This class provides methods to create different types of LLM instances.
    """
    
    @staticmethod
    def create(provider: str, model_name: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> BaseLLM:
        """
        Create an LLM instance based on the provider.
        
        Args:
            provider: LLM provider name (e.g., 'openai', 'anthropic', 'openrouter')
            model_name: Optional model name (if None, will use the default for the provider)
            api_key: Optional API key (if None, will try to load from environment)
            **kwargs: Additional model-specific parameters
            
        Returns:
            An LLM instance
            
        Raises:
            ValueError: If the provider is not supported
        """
        provider = provider.lower()
        
        if provider == "openai":
            model = model_name or "qwen/qwen3-32b:free"
            return OpenAILLM(model_name=model, api_key=api_key, **kwargs)
        
        elif provider == "anthropic":
            model = model_name or "claude-3-sonnet-20240229"
            return AnthropicLLM(model_name=model, api_key=api_key, **kwargs)
        
        elif provider == "openrouter":
            model = model_name or "qwen/qwen3-32b:free"
            return OpenRouterLLM(model_name=model, api_key=api_key, **kwargs)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def get_default_model(provider: str) -> str:
        """
        Get the default model name for a provider.
        
        Args:
            provider: LLM provider name
            
        Returns:
            Default model name for the provider
            
        Raises:
            ValueError: If the provider is not supported
        """
        provider = provider.lower()
        
        if provider == "openai":
            return "openai/gpt-4o-mini"
        
        elif provider == "anthropic":
            return "claude-3-sonnet-20240229"
        
        elif provider == "openrouter":
            return "qwen/qwen3-32b:free"
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def get_supported_providers() -> List[str]:
        """
        Get a list of supported LLM providers.
        
        Returns:
            List of supported provider names
        """
        return ["openai", "anthropic", "openrouter"] 