"""
Prompt loader for FinBrief.
Loads prompts from docs/prompts.md for easy customization.
"""
import os
import re
from typing import Dict


class PromptLoader:
    """Load and manage prompts from external configuration file."""
    
    def __init__(self, prompts_file: str = None):
        """
        Initialize prompt loader.

        Args:
            prompts_file: Path to prompts file (default: config/prompts.md)
        """
        if prompts_file is None:
            # Default to config/prompts.md in project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.dirname(current_dir)  # Go up from utils to src
            project_root = os.path.dirname(src_dir)  # Go up from src to project root
            prompts_file = os.path.join(project_root, 'config', 'prompts.md')
        
        self.prompts_file = prompts_file
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from markdown file."""
        prompts = {}
        
        if not os.path.exists(self.prompts_file):
            print(f"Warning: Prompts file not found: {self.prompts_file}")
            return self._get_default_prompts()
        
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse prompts from markdown
            # Format: ### prompt_name\n```\nprompt content\n```
            pattern = r'###\s+(\w+)\s*\n```\s*\n(.*?)\n```'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for name, prompt_text in matches:
                prompts[name] = prompt_text.strip()
            
            if not prompts:
                print("Warning: No prompts found in file, using defaults")
                return self._get_default_prompts()
            
            return prompts
            
        except Exception as e:
            print(f"Error loading prompts: {e}")
            return self._get_default_prompts()
    
    def _get_default_prompts(self) -> Dict[str, str]:
        """Fallback default prompts if file loading fails."""
        return {
            'system_instructions_beginner': """You are an educational financial analyst helping students learn about investing.
Your goal is to explain company information in simple, clear language that a beginner investor can understand.""",
            'system_instructions_expert': """You are a financial analyst providing detailed company analysis.
Ground all claims in the provided context and cite sources appropriately.""",
            'base_template': """Context from SEC filings and financial data:
{context}

Based on the context above, {query}

Ensure all factual claims are grounded in the provided context.""",
            'company_summary_enhancement': """Write a clear, beginner-friendly summary of what {company_name} does.""",
            'company_summary_generation': """Write a beginner-friendly summary of what {company_name} ({ticker}) does based on the provided context.""",
            'investor_takeaway': """Explain what this means for a beginner investor.""",
        }
    
    def get(self, prompt_name: str, **kwargs) -> str:
        """
        Get a prompt by name and format with variables.
        
        Args:
            prompt_name: Name of the prompt
            **kwargs: Variables to substitute in the prompt
            
        Returns:
            Formatted prompt string
        """
        prompt = self.prompts.get(prompt_name, '')
        
        if not prompt:
            print(f"Warning: Prompt '{prompt_name}' not found")
            return ''
        
        # Substitute variables
        try:
            return prompt.format(**kwargs)
        except KeyError as e:
            print(f"Warning: Missing variable {e} for prompt '{prompt_name}'")
            return prompt
    
    def reload(self):
        """Reload prompts from file."""
        self.prompts = self._load_prompts()


# Global instance
_prompt_loader = None


def get_prompt_loader() -> PromptLoader:
    """Get global prompt loader instance."""
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader


def get_prompt(prompt_name: str, **kwargs) -> str:
    """
    Convenience function to get a prompt.
    
    Args:
        prompt_name: Name of the prompt
        **kwargs: Variables to substitute
        
    Returns:
        Formatted prompt string
    """
    return get_prompt_loader().get(prompt_name, **kwargs)


