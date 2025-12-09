"""
Handler for LLM models optimized for Apple Silicon (M1/M2) Macs.
Primary model: GPT-2 Medium (355M) for educational financial briefs.
Supports optional quantization.

Model Selection (per CS372 design):
- GPT-2 Medium (default): 355M params, good balance of quality and speed
- DistilGPT2: 82M params, fastest option for testing
- TinyLlama: 1.1B params, higher quality but slower
- Mistral-7B: 7B params, highest quality but requires significant resources
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import os
import gc
import platform


def get_optimal_device():
    """Get the best available device for the current system."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and platform.processor() == 'arm':
        # Apple Silicon (M1/M2/M3) - use MPS for GPU acceleration
        return "mps"
    else:
        return "cpu"


def clear_memory():
    """Clear GPU/MPS memory and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def check_bitsandbytes_available():
    """Check if bitsandbytes is available for quantization."""
    try:
        import bitsandbytes
        return True
    except ImportError:
        return False


class FinBriefModel:
    """
    Wrapper for LLM models used in FinBrief educational financial analysis.
    Optimized for Apple Silicon.
    """
    
    # Default model per CS372 design: GPT-2 Medium (355M params)
    MODEL_NAME = "gpt2-medium"
    
    # Available models with their HuggingFace identifiers
    AVAILABLE_MODELS = {
        "gpt2": "gpt2",  # 124M params - fast baseline
        "gpt2-medium": "gpt2-medium",  # 355M params - DEFAULT per design.md
        "gpt2-large": "gpt2-large",  # 774M params - higher quality
        "distilgpt2": "distilgpt2",  # 82M params - fastest
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B params
        "mistral": "bitext/Mistral-7B-Wealth_Management",  # 7B params - highest quality
    }
    
    # Alias for backward compatibility
    SMALL_MODELS = AVAILABLE_MODELS
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None, 
                 use_quantization: bool = False):
        """
        Initialize the model optimized for Apple Silicon.
        
        Args:
            model_name: Model name or alias (defaults to gpt2-medium per design.md)
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            use_quantization: Whether to use 4-bit quantization (for large models on limited RAM)
        """
        # Resolve model name from alias
        if model_name and model_name.lower() in self.AVAILABLE_MODELS:
            self.model_name = self.AVAILABLE_MODELS[model_name.lower()]
        else:
            self.model_name = model_name or self.MODEL_NAME
        
        # Determine if this is a small/efficient model
        self.is_small_model = self.model_name in ['gpt2', 'gpt2-medium', 'distilgpt2']
        self.is_large_model = 'mistral' in self.model_name.lower() or '7b' in self.model_name.lower()
        
        self.device = device or get_optimal_device()
        self.use_quantization = use_quantization and self.is_large_model
        
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")
        if self.use_quantization:
            print("Quantization: Enabled (4-bit) - optimized for 16GB RAM")
        
        # Clear any existing memory
        clear_memory()
        
        try:
            self._load_model()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to GPT-2 Medium...")
            self.model_name = "gpt2-medium"
            self.is_small_model = True
            self.use_quantization = False
            try:
                self._load_model()
                print("Fallback model loaded successfully")
            except Exception as e2:
                print(f"Error loading fallback: {e2}")
                print("Falling back to DistilGPT2...")
                self.model_name = "distilgpt2"
                self._load_model()
    
    def _load_model(self):
        """Load the model with optimizations for the current device."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if we should use quantization for large models
        if self.use_quantization and self.device == "cuda" and check_bitsandbytes_available():
            # Use 4-bit quantization on CUDA (bitsandbytes requires CUDA)
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("Using 4-bit quantization (bitsandbytes)")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        elif self.device == "mps":
            # Apple Silicon optimization
            if self.is_large_model:
                print("Note: Loading large model on MPS. This may use significant memory.")
                print("If you run out of memory, try: --model gpt2-medium")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if not self.is_small_model else torch.float32,
            )
            self.model = self.model.to(self.device)
        elif self.device == "cuda":
            # NVIDIA GPU optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            # CPU fallback
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
            )
            self.model = self.model.to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
    
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of NEW tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        # Truncate input to fit within model's context window
        # Use smaller context for memory efficiency
        max_input_length = 512  # Reduced for memory efficiency
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_input_length,
            padding=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for efficiency
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Clean up common GPT-2 artifacts
            generated_text = self._clean_generation(generated_text, prompt)
            
            return generated_text
            
        finally:
            # Clear intermediate tensors
            del inputs
            if 'outputs' in dir():
                del outputs
            clear_memory()
    
    def _clean_generation(self, text: str, prompt: str) -> str:
        """
        Clean up GPT-2 generation artifacts.
        
        GPT-2 without fine-tuning often echoes prompts and produces messy output.
        """
        import re
        
        # Common prompt phrases to remove
        prompt_phrases = [
            "You are an educational financial analyst",
            "Your goal is to explain company information",
            "Context from SEC filings",
            "Provide a student-friendly analysis",
            "Important: Use simple language",
            "This is educational content",
            "Analysis:",
        ]
        
        for phrase in prompt_phrases:
            if phrase in text:
                # Find and remove everything up to and including this phrase
                idx = text.find(phrase)
                if idx != -1:
                    # Try to find a natural break point after the phrase
                    end_idx = text.find('\n', idx + len(phrase))
                    if end_idx != -1:
                        text = text[end_idx:].strip()
                    else:
                        text = text[idx + len(phrase):].strip()
        
        # Remove SEC boilerplate patterns
        boilerplate_patterns = [
            r'Commission File Number[:\s]*[\d-]+',
            r'\|[\s\|]*\-+[\s\|]*',  # Table separators
            r'Exact name of Registrant',
            r'For the transition period',
            r'\[SEC Filing\]',
            r'\[Definition\]',
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up multiple newlines and spaces
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove lines that are mostly special characters
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            # Keep line if it has substantial alphabetic content
            alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
            if alpha_ratio > 0.3 or len(line.strip()) == 0:
                clean_lines.append(line)
        
        return '\n'.join(clean_lines).strip()
    
    def analyze_with_context(self, context: str, query: str, for_beginners: bool = True, system_instructions: str = None) -> str:
        """
        Generate analysis using RAG context.

        Args:
            context: Retrieved context with citations
            query: Analysis query
            for_beginners: If True, use student-friendly language (default)
            system_instructions: Optional system instructions (if None, uses default inline prompt)

        Returns:
            Generated analysis
        """
        # Truncate context if too long (keep most relevant parts)
        max_context_chars = 16000  # Reduced for memory efficiency
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n[Context truncated for processing...]"

        # Use provided system instructions or fall back to inline prompt
        if system_instructions:
            # Use provided instructions with context and query
            prompt = f"""{system_instructions}

Context:
{context}

Task:
{query}

Response:"""
        elif for_beginners:
            # Default beginner prompt (fallback)
            prompt = f"""You are an educational financial analyst helping students learn about investing.
Your goal is to explain company information in simple, clear language that a beginner investor can understand.

Context from SEC filings and financial data:
{context}

{query}

Provide a student-friendly analysis covering:
1. Company Summary: What does this company do? (in simple terms)
2. Key Risks: What could go wrong? (explain in plain language, rate as High/Medium/Low)
3. Opportunities: What could help the company grow?
4. What This Means: Help a new investor understand the key takeaways

Important: Use simple language. Avoid jargon. When you must use financial terms, briefly explain them.
This is educational content, not investment advice.

Analysis:"""
        else:
            prompt = f"""You are an expert financial analyst specializing in investment research.
Analyze the following information and provide a comprehensive investment analysis.

Context from SEC filings and news:
{context}

{query}

Provide analysis covering:
1. Key financial metrics and trends
2. Risk factors (with severity: High/Medium/Low)
3. Growth opportunities
4. Summary of findings

Analysis:"""
        
        return self.generate(prompt, max_new_tokens=400, temperature=0.7)
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        clear_memory()


# Backward compatibility alias
MistralWealthManagementModel = FinBriefModel
