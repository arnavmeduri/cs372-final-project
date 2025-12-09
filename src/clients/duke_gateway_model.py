"""
Duke AI Gateway Model Handler

Wrapper for Duke's LiteLLM Gateway API to access frontier models
(GPT 4.1, GPT-5, Mistral, Llama 4, etc.) for FinBrief.

This replaces local model inference with cloud-hosted models for
better quality and performance.
"""
import os
from typing import Optional
from dotenv import load_dotenv
from ..utils.prompt_loader import get_prompt

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "openai package required for Duke AI Gateway. Install with: pip install openai>=1.0.0"
    )

load_dotenv(override=True)  # Override system env vars with .env file

# Duke AI Gateway configuration
DUKE_GATEWAY_URL = "https://litellm.oit.duke.edu/v1"
DUKE_GATEWAY_ENDPOINT = f"{DUKE_GATEWAY_URL}/responses"

# Available models on Duke Gateway with their context window limits
MODEL_CONTEXT_LIMITS = {
    'GPT 4.1': 128000,  # 128K tokens = ~512K chars
    'GPT 4.1 Mini': 128000,
    'GPT 4.1 Nano': 128000,
    'GPT 4o': 128000,
    'o4 Mini': 128000,
    'Llama 3.3': 128000,
    'Llama 4 Scout': 128000,
    'Llama 4 Maverick': 128000,
    'Mistral on-site': 32000,  # Smaller context window
    'gpt-5': 200000,  # Future models
    'gpt-5-chat': 200000,
    'gpt-5-mini': 128000,
    'gpt-5-nano': 128000,
    'gpt-oss-120b': 128000,
    'gpt-5-codex': 128000,
    'text-embedding-3-small': 8191
}

# Legacy list for backwards compatibility
AVAILABLE_MODELS = list(MODEL_CONTEXT_LIMITS.keys())


class DukeGatewayModel:
    """
    Model handler for Duke AI Gateway (LiteLLM).
    
    Provides the same interface as FinBriefModel but uses
    Duke's hosted frontier models instead of local inference.
    """
    
    def __init__(
        self,
        model_name: str = "GPT 4.1",
        api_token: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize Duke Gateway client.
        
        Args:
            model_name: Model to use (default: "GPT 4.1")
            api_token: LiteLLM token (or from LITELLM_TOKEN env var)
            verbose: Print status messages
        """
        self.verbose = verbose
        self.model_name = model_name
        
        # Get API token
        self.api_token = api_token or os.getenv('LITELLM_TOKEN')
        if not self.api_token:
            raise ValueError(
                "LITELLM_TOKEN not found. Set it in .env file or pass api_token parameter.\n"
                "Get your token from: https://dashboard.ai.duke.edu/"
            )
        
        # Validate model name
        if model_name not in AVAILABLE_MODELS:
            if self.verbose:
                print(f"Warning: '{model_name}' not in known models list. Attempting anyway...")
        
        # Initialize OpenAI client with Duke's endpoint
        print(f"[DUKE GATEWAY] Initializing client...")
        print(f"[DUKE GATEWAY] Base URL: {DUKE_GATEWAY_URL}")
        print(f"[DUKE GATEWAY] Model: {model_name}")
        print(f"[DUKE GATEWAY] Token: {self.api_token[:10]}...{self.api_token[-4:] if len(self.api_token) > 14 else '***'}")
        
        self.client = OpenAI(
            api_key=self.api_token,
            base_url=DUKE_GATEWAY_URL
        )
        
        print(f"[DUKE GATEWAY] Client initialized successfully")
        print(f"[DUKE GATEWAY] Ready to make API calls to Duke AI Gateway")
    
    def analyze_with_context(
        self,
        context: str,
        query: str,
        for_beginners: bool = True,
        system_instructions: str = None
    ) -> str:
        """
        Generate analysis using RAG context.

        Same interface as FinBriefModel.analyze_with_context()

        Args:
            context: Retrieved context with citations
            query: Analysis query
            for_beginners: If True, use student-friendly language
            system_instructions: Optional system instructions (if None, loads from default prompts)

        Returns:
            Generated analysis text
        """
        # Model-specific context window limits
        # Get token limit for this model (default to conservative 32K if unknown)
        token_limit = MODEL_CONTEXT_LIMITS.get(self.model_name, 32000)

        # Convert tokens to chars (rough estimate: 1 token ≈ 4 chars)
        # Use 80% of limit to leave room for response
        max_context_chars = int(token_limit * 4 * 0.8)

        print(f"[DUKE GATEWAY] Model '{self.model_name}' context limit: {token_limit:,} tokens (~{token_limit*4:,} chars)")
        print(f"[DUKE GATEWAY] Using 80% for input: {max_context_chars:,} chars max")

        if len(context) > max_context_chars:
            print(f"⚠️  [DUKE GATEWAY] Context truncated: {len(context):,} → {max_context_chars:,} chars")
            context = context[:max_context_chars] + "\n[Context truncated for processing...]"
        else:
            print(f"✅ [DUKE GATEWAY] Full context preserved: {len(context):,} chars (no truncation)")

        # Use provided system instructions or load from config
        if system_instructions:
            instructions = system_instructions
        else:
            # Load system instructions from config (fallback)
            if for_beginners:
                instructions = get_prompt('system_instructions')
            else:
                instructions = get_prompt('system_instructions_expert')

        # Fallback if prompt not found
        if not instructions:
            instructions = """You are an expert educational financial analyst. You help investors understand companies by extracting and explaining information from SEC filings and financial data.

CRITICAL RULES - YOU MUST FOLLOW EXACTLY:

1. OUTPUT MUST BE IN ENGLISH ONLY
   - Write ONLY in English language
   - Do NOT use Chinese, Arabic, or any non-English characters
   - Do NOT mix languages or include foreign text

2. BE FACTUAL AND ACCURATE
   - ONLY use information explicitly stated in the provided context
   - Do NOT hallucinate, invent, or fabricate any metrics, dates, or facts
   - If information is not in the context, state "not available" rather than guessing
   - Use specific numbers and facts ONLY if they appear in the context

3. WRITE CLEARLY AND PROFESSIONALLY
   - Use clear markdown formatting
   - Explain financial terms in plain language
   - Do NOT include citation markers like [1], [2] - the system handles references

QUALITY CHECK: Before finalizing, verify output is entirely in English with no foreign characters and all facts are sourced from context."""

        # Build input text
        # If custom system instructions were provided, use simple input format
        # Otherwise use base_template from config
        if system_instructions:
            # Custom instructions provided - use simple input format
            input_text = f"""Context:
{context}

Task:
{query}"""
        else:
            # No custom instructions - use base_template from config
            input_text = get_prompt('base_template', context=context, query=query)
        
        try:
            # Call Duke Gateway API
            print(f"\n{'='*80}")
            print(f"[DUKE GATEWAY] 🔄 Making API call to Duke Gateway...")
            print(f"{'='*80}")
            print(f"[DUKE GATEWAY] Model: {self.model_name}")
            print(f"[DUKE GATEWAY] Context length: {len(context)} characters (~{len(context)//4} tokens)")
            print(f"[DUKE GATEWAY] Query length: {len(query)} characters")
            print(f"[DUKE GATEWAY] Total input: {len(input_text)} characters (~{len(input_text)//4} tokens)")
            
            # Show context preview (first and last to detect truncation)
            print(f"\n[DUKE GATEWAY] Context preview (first 500 chars):")
            print(f"{'-'*80}")
            print(context[:500])
            print(f"[... middle content ...]")
            print(f"\n[DUKE GATEWAY] Context preview (last 500 chars):")
            print(context[-500:])
            print(f"{'-'*80}")

            print(f"\n[DUKE GATEWAY] Prompt/Query:")
            print(f"{'-'*80}")
            print(query[:500] if len(query) > 500 else query)
            if len(query) > 500:
                print(f"[... {len(query) - 500} more characters ...]")
                print(f"\n[Last 300 chars of prompt]:")
                print(query[-300:])
            print(f"{'-'*80}")
            
            print(f"\n[DUKE GATEWAY] Sending request...")
            print(f"  Model: {self.model_name}")

            # Duke Gateway API call
            # NOTE: Duke Gateway's responses.create() only accepts: model, instructions, input
            # Parameters like temperature, max_tokens, top_p are NOT supported
            response = self.client.responses.create(
                model=self.model_name,
                instructions=instructions,
                input=input_text
            )

            print(f"\n[DUKE GATEWAY] ✅ API call successful!")
            print(f"{'='*80}")

            # Verify response structure
            if hasattr(response, 'model') and response.model:
                print(f"[DUKE GATEWAY] Confirmed model used: {response.model}")
            else:
                print(f"[DUKE GATEWAY] ⚠️  Response did not include model confirmation")
            
            # Extract response text
            # Duke Gateway format: response.output[0].content[0].text
            generated_text = response.output[0].content[0].text
            print(f"[DUKE GATEWAY] Response length: {len(generated_text)} characters (~{len(generated_text)//4} tokens)")
            print(f"[DUKE GATEWAY] Response word count: ~{len(generated_text.split())} words")
            
            # Show response preview
            print(f"\n[DUKE GATEWAY] Response preview (first 800 chars):")
            print(f"{'-'*80}")
            print(generated_text[:800])
            print(f"[... {max(0, len(generated_text) - 800)} more characters ...]")
            print(f"{'-'*80}\n")
            
            # Clean up response
            generated_text = self._clean_response(generated_text)
            
            print(f"[DUKE GATEWAY] Analysis complete (using Duke Gateway)")
            print(f"{'='*80}\n")
            return generated_text
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                raise ValueError(
                    "Invalid LITELLM_TOKEN. Please verify your token at https://dashboard.ai.duke.edu/"
                )
            elif "budget" in error_msg.lower() or "budget_exceeded" in error_msg.lower():
                raise RuntimeError(
                    "Duke AI Gateway budget exceeded. This affects ALL models (including Mistral on-site).\n"
                    "Solutions:\n"
                    "  1. Wait for budget reset (check https://dashboard.ai.duke.edu/)\n"
                    "  2. Request budget increase from Duke IT\n"
                    "  3. Use local model: python -m src.finbrief TICKER --no-duke-gateway"
                )
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                raise RuntimeError(
                    "Duke AI Gateway rate limit exceeded. Please wait a moment and try again."
                )
            elif "404" in error_msg or "not found" in error_msg.lower():
                raise ValueError(
                    f"Model '{self.model_name}' not available. Check available models at https://dashboard.ai.duke.edu/"
                )
            else:
                raise RuntimeError(f"Duke AI Gateway error: {error_msg}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text from a prompt.
        
        Same interface as FinBriefModel.generate()
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate (hint for model)
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        instructions = "You are a helpful AI assistant providing clear, accurate information."
        
        try:
            print(f"[DUKE GATEWAY] 🔄 Making API call to Duke Gateway...")
            print(f"[DUKE GATEWAY] Model: {self.model_name}")
            print(f"[DUKE GATEWAY] Prompt length: {len(prompt)} characters")
            
            response = self.client.responses.create(
                model=self.model_name,
                instructions=instructions,
                input=prompt
            )
            
            print(f"[DUKE GATEWAY] ✅ API call successful!")
            
            generated_text = response.output[0].content[0].text
            print(f"[DUKE GATEWAY] Response length: {len(generated_text)} characters")
            
            generated_text = self._clean_response(generated_text)
            
            print(f"[DUKE GATEWAY] Generation complete (using Duke Gateway)")
            return generated_text
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                raise ValueError("Invalid LITELLM_TOKEN")
            elif "budget" in error_msg.lower() or "budget_exceeded" in error_msg.lower():
                raise RuntimeError(
                    "Duke AI Gateway budget exceeded. This affects ALL models.\n"
                    "Use --no-duke-gateway flag to use local model instead."
                )
            elif "429" in error_msg:
                raise RuntimeError("Rate limit exceeded")
            else:
                raise RuntimeError(f"Duke AI Gateway error: {error_msg}")
    
    def _clean_response(self, text: str) -> str:
        """
        Clean up model response text.

        Args:
            text: Raw response text

        Returns:
            Cleaned text
        """
        import re

        # Remove common artifacts
        text = text.strip()

        # Remove Mistral on-site specific disclaimers/garbage
        mistral_patterns = [
            r'\{I am a creativity and language model[^}]*\}',
            r'NOTE IF YOU LEGITREAILY FOUND ME HELPFUL[^\n]*',
            r'PLEASE REMEMBER TO TIP ME[^\n]*',
            r'THE LIVING ISN\'T EASY DICE ROLLIN\' BEHIND THIS SCREEN[^\n]*',
            r'\{Disclaimer[^}]*\}',
            r'I am an AI[^\n]*do not have personal experiences[^\n]*',
        ]

        for pattern in mistral_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        # Final cleanup
        text = text.strip()

        return text
    
    @staticmethod
    def is_available() -> bool:
        """
        Check if Duke Gateway is available (token configured).
        
        Returns:
            True if LITELLM_TOKEN is set
        """
        return os.getenv('LITELLM_TOKEN') is not None
    
    @staticmethod
    def list_available_models() -> list:
        """
        Get list of available models.
        
        Returns:
            List of model names
        """
        return AVAILABLE_MODELS.copy()

