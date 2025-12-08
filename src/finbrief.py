"""
FinBrief: Educational Financial Brief Generator

Main application that integrates all components to generate student-oriented
investment education briefs.

Components:
- SEC EDGAR API (10-K, 10-Q filings)
- Definitions Corpus (Investor.gov, Investopedia)
- Finnhub API (financial metrics)
- RAG System (MiniLM + FAISS)
- GPT-2 Medium Generator

Usage:
    from src.finbrief import FinBriefApp
    
    app = FinBriefApp()
    brief = app.generate_brief("AAPL")
    print(app.format_brief(brief))
"""
import os
import re
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from dotenv import load_dotenv

from .sec_edgar_client import SECEdgarClient as SECEdgarClientOld
from .edgartools_client import EdgarToolsClient
from .finnhub_client import FinnhubClient, FinancialMetrics
from .rag_system import RAGSystem
from .model_handler import FinBriefModel, clear_memory
from .duke_gateway_model import DukeGatewayModel
from .prompt_loader import get_prompt
from .educational_brief import (
    EducationalBrief, EducationalBriefFormatter,
    RiskItem, OpportunityItem, TermExplanation, FinancialMetric, Citation,
    DifficultyLevel, RiskSeverity
)
from .rich_formatter import RichAnalysisFormatter
from .confidence_head import HeuristicConfidenceEstimator

load_dotenv()


class FinBriefApp:
    """
    Main FinBrief application for generating educational investment briefs.
    """
    
    # Company name mappings
    COMPANY_NAMES = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.',
        'META': 'Meta Platforms Inc.',
        'TSLA': 'Tesla Inc.',
        'NVDA': 'NVIDIA Corporation',
        'IBM': 'IBM Corporation',
        'JPM': 'JPMorgan Chase & Co.',
        'V': 'Visa Inc.',
        'JNJ': 'Johnson & Johnson',
        'NFLX': 'Netflix Inc.',
        'INTC': 'Intel Corporation',
        'AMD': 'Advanced Micro Devices Inc.',
    }
    
    def __init__(
        self,
        finnhub_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        use_finnhub: bool = True,
        use_duke_gateway: Optional[bool] = None,
        duke_model: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize FinBrief application.
        
        Args:
            finnhub_api_key: Finnhub API key (or from FINNHUB_API_KEY env var)
            model_name: LLM model to use (default: gpt2-medium) - only used if not using Duke Gateway
            use_finnhub: Whether to fetch Finnhub metrics
            use_duke_gateway: Whether to use Duke AI Gateway (auto-detect if None)
            duke_model: Duke Gateway model name (default: GPT 4.1)
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.use_finnhub = use_finnhub
        self.model_name = model_name
        
        # Duke Gateway configuration
        # Auto-detect: use gateway if token is available
        print(f"[FINBRIEF INIT] Checking Duke Gateway availability...")
        duke_available = DukeGatewayModel.is_available()
        print(f"[FINBRIEF INIT] Duke Gateway available: {duke_available}")
        
        if use_duke_gateway is None:
            self.use_duke_gateway = duke_available
            print(f"[FINBRIEF INIT] Auto-detection: use_duke_gateway = {self.use_duke_gateway}")
        else:
            self.use_duke_gateway = use_duke_gateway
            print(f"[FINBRIEF INIT] Explicit setting: use_duke_gateway = {self.use_duke_gateway}")
        
        self.duke_model = duke_model or os.getenv('DUKE_AI_MODEL', 'GPT 4.1')
        print(f"[FINBRIEF INIT] Duke model: {self.duke_model}")
        
        self._log("Initializing FinBrief...")
        
        # Initialize SEC EDGAR client (use edgartools for better section extraction)
        self._log("Loading SEC EDGAR client (edgartools)...")
        name = os.getenv('SEC_EDGAR_NAME', 'Student')
        email = os.getenv('SEC_EDGAR_EMAIL', 'student@university.edu')

        try:
            # Try edgartools first (10x better section extraction)
            self.sec_client = EdgarToolsClient(
                name=name,
                email=email
            )
            self._log("✅ Using edgartools (better section extraction)")
        except Exception as e:
            # Fallback to old client if edgartools unavailable
            self._log(f"⚠️  edgartools unavailable, using legacy client: {e}")
            company_name = os.getenv('SEC_EDGAR_COMPANY_NAME', 'FinBriefApp')
            self.sec_client = SECEdgarClientOld(
                company_name=company_name,
                name=name,
                email=email
            )
        
        # Initialize Finnhub client (optional)
        self.finnhub_client = None
        if use_finnhub:
            api_key = finnhub_api_key or os.getenv('FINNHUB_API_KEY')
            if api_key:
                try:
                    self.finnhub_client = FinnhubClient(api_key=api_key)
                    self._log("Finnhub API connected")
                except Exception as e:
                    self._log(f"Warning: Could not initialize Finnhub: {e}")
            else:
                self._log("Finnhub API key not provided. Metrics will be unavailable.")
        
        # Initialize RAG system
        self._log("Loading RAG system (MiniLM embeddings)...")
        self.rag = RAGSystem()
        
        # Formatter
        self.formatter = EducationalBriefFormatter()
        self.rich_formatter = RichAnalysisFormatter()
        
        # Model (lazy loaded)
        self.model = None
        
        # Confidence estimator
        self.confidence_estimator = HeuristicConfidenceEstimator()
        
        self._log("FinBrief initialized successfully!")
    
    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[FinBrief] {message}")
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name from ticker."""
        return self.COMPANY_NAMES.get(ticker.upper(), f"{ticker.upper()} Corporation")
    
    def _load_model(self):
        """
        Lazy load the LLM model with priority:
        1. Duke AI Gateway (if enabled and token available)
        2. LoRA adapter (if exists)
        3. Local model (TinyLlama or specified)
        """
        if self.model is None:
            # Priority 1: Duke AI Gateway
            print(f"[MODEL LOADER] Checking model options...")
            print(f"[MODEL LOADER] use_duke_gateway: {self.use_duke_gateway}")
            
            if self.use_duke_gateway:
                print(f"[MODEL LOADER] Attempting to load Duke AI Gateway...")
                try:
                    is_available = DukeGatewayModel.is_available()
                    print(f"[MODEL LOADER] Duke Gateway available: {is_available}")
                    
                    if is_available:
                        print(f"[MODEL LOADER] Using Duke AI Gateway")
                        print(f"[MODEL LOADER] Model: {self.duke_model}")
                        self._log(f"Loading Duke AI Gateway model: {self.duke_model}")
                        self.model = DukeGatewayModel(
                            model_name=self.duke_model,
                            verbose=True
                        )
                        print(f"[MODEL LOADER] Duke Gateway model loaded successfully")
                        print(f"[MODEL LOADER] Model type: {type(self.model).__name__}")
                        return
                    else:
                        print(f"[MODEL LOADER] Duke Gateway requested but LITELLM_TOKEN not found")
                        if self.verbose:
                            self._log("Duke Gateway requested but LITELLM_TOKEN not found. Falling back to local model.")
                except Exception as e:
                    print(f"[MODEL LOADER] Failed to load Duke Gateway: {e}")
                    import traceback
                    traceback.print_exc()
                    if self.verbose:
                        self._log(f"Failed to load Duke Gateway: {e}. Falling back to local model.")
                    # Continue to fallback options
            else:
                print(f"[MODEL LOADER] Duke Gateway not requested (use_duke_gateway=False)")
            
            # Priority 2: Check for fine-tuned LoRA adapter
            # Path resolution: from src/finbrief.py -> project_root/models/lora_adapter
            src_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(src_dir)
            lora_path = os.path.join(project_root, 'models', 'lora_adapter')
            adapter_config_path = os.path.join(lora_path, 'adapter_config.json')
            
            if os.path.exists(adapter_config_path):
                # Load the adapter config to get the base model name
                print(f"[MODEL LOADER] Duke Gateway not used, falling back to LoRA adapter")
                import json
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model = adapter_config.get('base_model_name_or_path', 'gpt2-medium')
                self._log(f"Loading fine-tuned model: {base_model} + LoRA adapter")
                self.model = FinBriefModel(model_name=base_model, lora_path=lora_path)
                print(f"[MODEL LOADER] LoRA model loaded (NOT using Duke Gateway)")
            else:
                print(f"[MODEL LOADER] Duke Gateway not used, falling back to local model")
                if self.model_name:
                    model_name = self.model_name
                else:
                    # Map to full HuggingFace path
                    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                self._log(f"Loading local instruction-tuned model: {model_name}")
                self.model = FinBriefModel(model_name=model_name)
                print(f"[MODEL LOADER] Local model loaded (NOT using Duke Gateway)")
                print(f"[MODEL LOADER] Model type: {type(self.model).__name__}")
    
    def fetch_sec_filings(self, ticker: str, filing_type: str = "10-K", limit: int = 1) -> List[Dict]:
        """
        Fetch SEC filings for a company.
        
        Args:
            ticker: Stock ticker symbol
            filing_type: Type of filing (10-K, 10-Q, 8-K)
            limit: Number of filings to fetch
            
        Returns:
            List of filing dictionaries
        """
        self._log(f"Fetching {filing_type} filings for {ticker}...")
        
        if filing_type == "10-K":
            filings = self.sec_client.get_annual_filings(ticker, limit=limit)
        elif filing_type == "10-Q":
            filings = self.sec_client.get_quarterly_filings(ticker, limit=limit)
        elif filing_type == "8-K":
            filings = self.sec_client.get_material_event_filings(ticker, limit=limit)
        else:
            filings = self.sec_client.get_filings(ticker, form_type=filing_type, limit=limit)
        
        self._log(f"Retrieved {len(filings)} {filing_type} filings")
        return filings
    
    def fetch_metrics(self, ticker: str) -> Optional[FinancialMetrics]:
        """
        Fetch financial metrics from Finnhub.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            FinancialMetrics object or None
        """
        if not self.finnhub_client:
            self._log("Finnhub not available. Skipping metrics.")
            return None
        
        self._log(f"Fetching Finnhub metrics for {ticker}...")
        try:
            metrics = self.finnhub_client.get_metrics_for_brief(ticker)
            if metrics:
                self._log("Metrics retrieved successfully")
            return metrics
        except Exception as e:
            self._log(f"Error fetching metrics: {e}")
            return None
    
    def _extract_risks_from_content(self, content: str, citations: List[Dict]) -> List[RiskItem]:
        """Extract risk items from SEC filing content."""
        risks = []
        seen_descriptions = set()
        
        # Clean content first
        clean_content = re.sub(r'<[^>]+>', '', content)
        clean_content = re.sub(r'http[s]?://\S+', '', clean_content)
        clean_content = re.sub(r'\[Definition\]', '', clean_content)
        
        # Look for risk-related sentences
        sentences = re.split(r'[.!?]\s+', clean_content)
        risk_keywords = ['risk', 'could adversely', 'may negatively', 'uncertainty', 
                        'challenge', 'threat', 'competition', 'regulatory', 'litigation',
                        'decline', 'loss', 'adverse', 'failure']
        
        # Get the filing citation
        filing_citation = "(SEC Filing)"
        for c in citations:
            if c.get('source_type') == 'sec_filing':
                filing_citation = f"(10-Q/10-K Filing)"
                break
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_lower = sentence.lower()
            
            # Skip if already seen similar content
            key = sentence_lower[:50]
            if key in seen_descriptions:
                continue
            
            # Skip definition chunks
            if 'example:' in sentence_lower or sentence.startswith('Example'):
                continue
                
            # Must contain risk keyword and be substantive
            if any(kw in sentence_lower for kw in risk_keywords):
                if len(sentence) > 50 and len(sentence) < 250:
                    # Skip if mostly numbers or special chars
                    if sum(c.isalpha() for c in sentence) < len(sentence) * 0.5:
                        continue
                    
                    # Determine severity
                    if any(w in sentence_lower for w in ['significant', 'material', 'substantial', 'major']):
                        severity = RiskSeverity.HIGH
                    elif any(w in sentence_lower for w in ['could', 'may', 'potential']):
                        severity = RiskSeverity.MEDIUM
                    else:
                        severity = RiskSeverity.MEDIUM
                    
                    seen_descriptions.add(key)
                    risks.append(RiskItem(
                        description=sentence[:200],
                        severity=severity,
                        citation=filing_citation
                    ))
                    
                    if len(risks) >= 5:
                        break
        
        return risks
    
    def _extract_opportunities_from_content(self, content: str, citations: List[Dict]) -> List[OpportunityItem]:
        """Extract opportunity items from SEC filing content."""
        opportunities = []
        seen_descriptions = set()
        
        # Clean content first
        clean_content = re.sub(r'<[^>]+>', '', content)
        clean_content = re.sub(r'http[s]?://\S+', '', clean_content)
        clean_content = re.sub(r'\[Definition\]', '', clean_content)
        
        sentences = re.split(r'[.!?]\s+', clean_content)
        opp_keywords = ['growth', 'expansion', 'opportunity', 'increase', 'innovation',
                       'new market', 'strong demand', 'competitive advantage', 'improved',
                       'successful', 'positive', 'momentum']
        
        # Get the filing citation
        filing_citation = "(SEC Filing)"
        for c in citations:
            if c.get('source_type') == 'sec_filing':
                filing_citation = f"(10-Q/10-K Filing)"
                break
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_lower = sentence.lower()
            
            # Skip if already seen similar content
            key = sentence_lower[:50]
            if key in seen_descriptions:
                continue
            
            # Skip definition chunks
            if 'example:' in sentence_lower or sentence.startswith('Example'):
                continue
            
            if any(kw in sentence_lower for kw in opp_keywords):
                if len(sentence) > 50 and len(sentence) < 250:
                    # Skip if mostly numbers or special chars
                    if sum(c.isalpha() for c in sentence) < len(sentence) * 0.5:
                        continue
                    
                    # Categorize
                    if 'revenue' in sentence_lower or 'sales' in sentence_lower:
                        category = "Revenue Growth"
                    elif 'market' in sentence_lower:
                        category = "Market Expansion"
                    elif 'innovation' in sentence_lower or 'product' in sentence_lower or 'technology' in sentence_lower:
                        category = "Product Innovation"
                    elif 'cloud' in sentence_lower or 'ai' in sentence_lower or 'digital' in sentence_lower:
                        category = "Technology"
                    else:
                        category = "Strategic Opportunity"
                    
                    seen_descriptions.add(key)
                    opportunities.append(OpportunityItem(
                        description=sentence[:200],
                        citation=filing_citation,
                        category=category
                    ))
                    
                    if len(opportunities) >= 5:
                        break
        
        return opportunities
    
    def _get_company_summary(self, ticker: str, company_name: str, filing_content: str) -> str:
        """
        Get company summary from SEC filing - generalized extraction that works for ANY company.
        Uses universal SEC 10-K patterns, no company-specific logic.
        """
        import re
        
        # Get first word of company name for pattern matching
        company_first_word = company_name.split()[0] if company_name else ticker
        
        # Skip forward-looking statements and risk factors (common false positives)
        # Business description is typically between table of contents and Item 1A (Risk Factors)
        # "Item 1. Business" in TOC is just a reference - actual content comes after
        item1a_idx = filing_content.lower().find('item 1a')
        
        if item1a_idx > 0:
            # Search content before Item 1A (Risk Factors)
            # Skip first 2000 chars (usually table of contents and headers)
            search_start = min(2000, len(filing_content) // 4)
            search_content = filing_content[search_start:item1a_idx]
        else:
            # No Item 1A found - search from after headers
            search_start = min(2000, len(filing_content) // 4)
            search_content = filing_content[search_start:]
        
        # Universal patterns - work for any SEC 10-K filing
        # Order matters: more specific patterns first
        patterns = [
            # Pattern 1: "Company Background The Company..." (very common - Apple style)
            r'Company\s+Background\s+(The\s+Company\s+[^.]+\.[^.]+\.)',
            
            # Pattern 2: "General We operate/are/provide..." (Costco style - "General" as section header)
            # Must be followed by "We" to avoid forward-looking statements
            r'\bGeneral\s+(We\s+(?:operate|are|provide|design|manufacture|develop|sell)[^.]{30,500}\.)',
            
            # Pattern 3: "GENERAL [Company] is a..." (some companies)
            r'\bGENERAL\s+([A-Z][A-Za-z]+\s+is\s+a\s+[^.]+\.[^.]+\.)',
            
            # Pattern 4: "The Company designs/manufactures/provides/operates..."
            r'(The\s+Company\s+(?:designs?|manufactures?|provides?|develops?|is\s+a|operates?|offers?|pioneers?)[^.]+\.[^.]+\.)',
            
            # Pattern 5: "We are a/operate/provide..." (first person, very common)
            # Must start sentence to avoid mid-sentence matches
            r'(^|\n|\.\s+)(We\s+(?:are\s+a|operate|provide|design|manufacture|develop|sell)\s+[^.]+\.[^.]+\.)',
            
            # Pattern 6: "[Company Name] is a/is now a..." (any company)
            rf'({re.escape(company_first_word)}\s+(?:is\s+(?:a|an|now\s+a|the)|pioneered?|provides?)[^.]+\.[^.]+\.)',
            
            # Pattern 7: "Our mission/business is..."
            r'(Our\s+(?:mission|business)\s+is\s+[^.]+\.[^.]+\.)',
            
            # Pattern 8: "Business Overview" section
            r'Business\s+Overview\s+([A-Z][^.]+\.[^.]+\.)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, search_content, re.DOTALL | re.IGNORECASE | re.MULTILINE)
            if match:
                # Get the captured group (group 1 or 2 depending on pattern)
                raw_text = match.group(1) if match.lastindex >= 1 else match.group(0)
                raw_text = raw_text.strip()
                
                # Skip if it's a forward-looking statement, boilerplate, or form fields
                skip_words = ['forward-looking', 'may relate', 'uncertainties', 'risks', 
                             'commission file', 'registrant', 'pursuant to', 'table of contents',
                             'accelerated filer', 'emerging growth company', 'check mark',
                             'section 13', 'exchange act', '☒', '☐', '☑']
                if any(word in raw_text.lower() for word in skip_words):
                    continue
                
                # Skip if it's mostly form fields (has checkboxes or too many special chars)
                if raw_text.count('☒') + raw_text.count('☐') + raw_text.count('☑') > 0:
                    continue
                
                summary = self._extract_clean_sentences(raw_text, max_sentences=2)
                if summary and len(summary) > 80:
                    return summary
        
        # Fallback: Use SEC client's extract_section for Item 1
        item_1_content = self.sec_client.extract_section(filing_content, 'item_1')
        if item_1_content:
            # Extract first good sentence from Item 1
            sentences = re.split(r'(?<=[.!?])\s+', item_1_content[:2000])
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 60 and len(sentence) < 400 and
                    sentence[0].isupper() and
                    not any(word in sentence.lower() for word in ['item ', 'table of', 'forward-looking', 'risks'])):
                    text = self._clean_extracted_text(sentence)
                    if len(text) > 60:
                        return text
        
        # Last fallback: Find first substantive business description sentence
        # Look for sentences with business action verbs
        business_verbs = r'(?:designs?|manufactures?|provides?|develops?|operates?|offers?|sells?|pioneered?|creates?|are\s+a)'
        sentences = re.split(r'(?<=[.!?])\s+', search_content[:15000])  # First 15k chars
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Must have business verb and be substantive
            skip_indicators = ['forward-looking', 'may relate', 'uncertainties', 'risks', 
                             'item ', 'table of', 'accelerated filer', 'emerging growth',
                             'check mark', 'pursuant to', '☒', '☐', '☑', 'commission file',
                             'affiliates', 'calculation', 'determination']
            if (re.search(business_verbs, sentence, re.IGNORECASE) and 
                len(sentence) > 50 and len(sentence) < 400 and
                sentence[0].isupper() and
                not any(word in sentence.lower() for word in skip_indicators)):
                text = self._clean_extracted_text(sentence)
                if len(text) > 60:
                    return text
        
        return f"Business description for {company_name} - see SEC filing Item 1 for details."
    
    def _extract_full_sentences(self, text: str, max_length: int = 300) -> str:
        """
        Extract full sentences from text up to max_length characters.
        Ensures we don't cut off mid-sentence.
        
        Args:
            text: Source text to extract from
            max_length: Maximum character length for excerpt
            
        Returns:
            Excerpt with complete sentences, truncated if necessary
        """
        if not text:
            return ""
        
        # Clean up newlines
        text = text.replace('\n', ' ').strip()
        
        if len(text) <= max_length:
            return text
        
        # Take first max_length chars
        truncated = text[:max_length]
        
        # Find last complete sentence (period, exclamation, question mark followed by space)
        sentence_endings = ['. ', '! ', '? ']
        last_sentence_end = -1
        
        # Search backwards from the end, but at least 50% into the text
        search_start = max(0, max_length - 100)
        for i in range(len(truncated) - 1, search_start, -1):
            if i < len(truncated) - 1:
                # Check for sentence ending followed by space
                if truncated[i] in ['.', '!', '?']:
                    if i + 1 < len(truncated) and truncated[i + 1] in [' ', '\t']:
                        last_sentence_end = i + 1
                        break
        
        if last_sentence_end > max_length * 0.5:  # Found a sentence boundary at least 50% through
            excerpt = truncated[:last_sentence_end].strip()
        else:
            # No sentence boundary found, find last complete word
            last_space = truncated.rfind(' ', 0, max_length)
            if last_space > max_length * 0.7:  # Only use if we got most of the length
                excerpt = truncated[:last_space].strip()
            else:
                # Fallback: just truncate at max_length
                excerpt = truncated.strip()
        
        # Add ellipsis if we truncated
        if len(text) > max_length:
            excerpt += "..."
        
        return excerpt
    
    def _extract_clean_sentences(self, text: str, max_sentences: int = 3) -> str:
        """Extract clean, complete sentences from text."""
        import re
        
        # Clean XBRL/HTML artifacts
        text = self._clean_extracted_text(text)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        clean_sentences = []
        for s in sentences[:max_sentences]:
            s = s.strip()
            # Must be substantive (not just a header or fragment)
            if len(s) > 40 and s[0].isupper() and '.' in s:
                # Skip table of contents entries
                if not re.match(r'^Item\s+\d', s, re.IGNORECASE):
                    clean_sentences.append(s)
        
        return ' '.join(clean_sentences)
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text from SEC filing."""
        import re
        # Remove XBRL artifacts
        text = re.sub(r'\b[a-z]{2,10}:[A-Za-z0-9]+\b', '', text)
        text = re.sub(r'®|™|©', '', text)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _clean_sec_text(self, text: str, max_sentences: int = 4) -> str:
        """
        Aggressively clean SEC filing text to remove XBRL/XML artifacts.
        """
        if not text:
            return ""
        
        # Remove all XML/HTML tags
        clean = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove XBRL namespace patterns
        clean = re.sub(r'\b[a-z]{2,10}:[A-Za-z0-9]+\b', '', clean)  # xbrli:shares, us-gaap:Revenue
        clean = re.sub(r'\biso\d+:[A-Z]+\b', '', clean)  # iso4217:USD
        clean = re.sub(r'#[A-Za-z0-9_]+', '', clean)  # Fragment IDs
        
        # Remove dates and numbers that look like IDs
        clean = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', clean)  # Dates
        clean = re.sub(r'\b\d{10,}\b', '', clean)  # Long numbers (CIKs, etc)
        clean = re.sub(r'\b0{6,}\d+\b', '', clean)  # Zero-padded numbers
        
        # Remove common SEC boilerplate phrases
        boilerplate = [
            r'for the transition period from.*?to',
            r'commission file number[:\s]*[\d-]+',
            r'exact name of registrant as specified',
            r'state or other jurisdiction',
            r'i\.r\.s\. employer identification',
            r'address of principal executive',
            r'registrant.s telephone number',
            r'securities registered pursuant',
            r'indicate by check mark',
            r'large accelerated filer',
            r'smaller reporting company',
            r'emerging growth company',
        ]
        for pattern in boilerplate:
            clean = re.sub(pattern, '', clean, flags=re.IGNORECASE)
        
        # Remove table formatting artifacts
        clean = re.sub(r'\|+', ' ', clean)
        clean = re.sub(r'-{3,}', ' ', clean)
        clean = re.sub(r'_{3,}', ' ', clean)
        clean = re.sub(r'☒|☐', '', clean)
        
        # Clean whitespace
        clean = re.sub(r'[\r\n\t]+', ' ', clean)
        clean = re.sub(r'\s{2,}', ' ', clean)
        clean = clean.strip()
        
        # Extract meaningful sentences
        sentences = re.split(r'(?<=[.!?])\s+', clean)
        good_sentences = []
        
        for s in sentences:
            s = s.strip()
            
            # Skip too short or too long
            if len(s) < 40 or len(s) > 400:
                continue
            
            # Must start with a letter
            if not s or not s[0].isalpha():
                continue
            
            # Must have high alphabetic ratio (not tables/numbers)
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in s) / len(s)
            if alpha_ratio < 0.75:
                continue
            
            # Skip if contains XBRL-like patterns
            if re.search(r'[a-z]+:[A-Z]', s) or 'fasb.org' in s.lower():
                continue
            
            # Skip header-only sentences
            if s.lower().startswith(('item ', 'part ', 'table of')):
                continue
            
            good_sentences.append(s)
            if len(good_sentences) >= max_sentences:
                break
        
        return ' '.join(good_sentences)
    
    def _is_valid_summary(self, text: str) -> bool:
        """
        Check if generated text is a valid summary (not garbage).
        
        GPT-2 (even with LoRA) often produces:
        - Echoed prompts
        - Raw SEC boilerplate
        - Mostly special characters
        - Repetitive patterns
        """
        if not text or len(text) < 50:
            return False
        
        # Check for prompt echo and bad patterns
        bad_phrases = [
            "You are an educational",
            "Your goal is to explain",
            "Context from SEC filings",
            "Commission File Number",
            "Exact name of Registrant",
            "transition period from",
            "---|---|---",
            "☒", "☐",
            "What does this company do?",  # Repetitive pattern from undertrained model
            "Company Summary:",  # Echo of prompt structure
        ]
        
        for phrase in bad_phrases:
            if phrase in text:
                return False
        
        # Check for repetition (undertrained LoRA symptom)
        words = text.lower().split()
        if len(words) > 10:
            # Check if any phrase repeats more than 3 times
            from collections import Counter
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            if trigram_counts and trigram_counts.most_common(1)[0][1] > 3:
                return False
        
        # Check if mostly alphabetic (not tables/numbers)
        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count < len(text) * 0.4:
            return False
        
        # Check for reasonable sentence structure
        sentences = text.split('.')
        if len(sentences) < 2:
            return False
        
        return True
    
    def _extract_clean_summary(self, content: str, company_name: str) -> str:
        """Extract a clean company summary from filing content."""
        # Remove XBRL/XML tags and artifacts (more aggressive)
        clean = re.sub(r'<[^>]+>', ' ', content)
        clean = re.sub(r'http[s]?://\S+', '', clean)
        clean = re.sub(r'\d{10}', '', clean)  # Remove CIK-like numbers
        clean = re.sub(r'\|+', ' ', clean)  # Remove table separators
        clean = re.sub(r'-{3,}', ' ', clean)  # Remove horizontal lines
        clean = re.sub(r'_{3,}', ' ', clean)  # Remove underlines
        clean = re.sub(r'[\r\n\t]+', ' ', clean)
        clean = re.sub(r'\s{2,}', ' ', clean)
        
        # Remove XBRL namespace patterns (critical!)
        clean = re.sub(r'[a-z]+:[A-Za-z]+\d*', '', clean)  # xbrli:shares, us-gaap:CommonStock, etc.
        clean = re.sub(r'iso\d+:[A-Z]+', '', clean)  # iso4217:USD
        clean = re.sub(r'aapl:[A-Za-z]+', '', clean)  # Company-specific XBRL
        clean = re.sub(r'\d{4}-\d{2}-\d{2}', '', clean)  # Dates like 2024-09-29
        clean = re.sub(r'#[A-Za-z]+', '', clean)  # Fragment identifiers
        clean = re.sub(r'\b[A-Z]{2,}[a-z]+[A-Z][a-z]+\b', '', clean)  # CamelCase XBRL terms
        
        # Skip common header/legal patterns
        skip_patterns = [
            r'united states securities',
            r'commission file number',
            r'exact name of registrant',
            r'state or other jurisdiction',
            r'i\.r\.s\.',
            r'false\d{4}',
            r'washington, d\.c\.',
            r'p\d+y',  # Duration patterns like P1Y
            r'indicate by check mark',
            r'registrant',
            r'accelerated filer',
            r'smaller reporting company',
            r'emerging growth company',
            r'exchange act',
            r'rule \d+',
            r'section \d+',
            r'yes ☒',
            r'no ☐',
            r'pursuant to',
            r'securities exchange',
        ]
        
        # Find meaningful sentences (not just numbers/codes)
        sentences = re.split(r'[.!?]\s+', clean)
        good_sentences = []
        
        for s in sentences:
            s = s.strip()
            s_lower = s.lower()
            
            # Skip if too short or too long
            if len(s) < 50 or len(s) > 300:
                continue
            
            # Skip if starts with non-letter
            if not s[0].isalpha():
                continue
            
            # Skip if too many numbers/special chars
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in s) / len(s)
            if alpha_ratio < 0.7:
                continue
            
            # Skip XBRL/legal artifacts
            if 'fasb.org' in s_lower or 'gaap' in s_lower:
                continue
            
            # Skip filing header patterns
            if any(re.search(pattern, s_lower) for pattern in skip_patterns):
                continue
            
            good_sentences.append(s)
            if len(good_sentences) >= 3:
                break
        
        # Return extracted content or empty string
        if good_sentences:
            return '. '.join(good_sentences) + '.'
        
        return ""
    
    def _determine_difficulty(self, content: str, ticker: str) -> Tuple[DifficultyLevel, str]:
        """Determine beginner difficulty level for understanding the company."""
        # Count complex financial terms
        complex_terms = ['derivatives', 'hedging', 'amortization', 'goodwill impairment',
                        'deferred tax', 'contingent consideration', 'variable interest entity']
        
        content_lower = content.lower()
        complexity_count = sum(1 for term in complex_terms if term in content_lower)
        
        # Large-cap well-known companies are generally easier
        well_known = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        if ticker.upper() in well_known and complexity_count < 3:
            return DifficultyLevel.EASY, "Well-known company with straightforward business model."
        elif complexity_count > 5:
            return DifficultyLevel.DIFFICULT, "Complex financial structure with advanced accounting terminology."
        else:
            return DifficultyLevel.MODERATE, "Standard complexity with some technical financial terms."
    
    def generate_brief(
        self,
        ticker: str,
        filing_type: str = "10-K",
        use_model: bool = True,
        use_rag: bool = True
    ) -> EducationalBrief:
        """
        Generate a complete educational brief for a company.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            filing_type: Type of SEC filing to analyze (10-K or 10-Q)
            use_model: Whether to use LLM for generation (slower but better)
            use_rag: Whether to use RAG (SEC filings + Finnhub) (default: True)
            
        Returns:
            EducationalBrief object
        """
        # Show mode clearly
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE ANALYSIS MODE")
        print(f"   This will produce 1000-2000 words of detailed analysis")
        
        if not use_rag:
            print(f"RAG DISABLED - Using only LLM general knowledge")
            print(f"   (No SEC filings, no Finnhub metrics)")
        else:
            print(f"RAG ENABLED - Using SEC filings + Finnhub metrics")
        
        print(f"{'='*80}\n")
        
        return self.generate_rich_analysis(ticker, filing_type, use_rag=use_rag)
        
        ticker = ticker.upper()
        company_name = self._get_company_name(ticker)
        
        self._log(f"Generating educational brief for {company_name} ({ticker})")
        
        # 1. Fetch SEC filings (if use_rag is True)
        if use_rag:
            filings = self.fetch_sec_filings(ticker, filing_type=filing_type, limit=1)
            
            if not filings:
                raise ValueError(f"Could not fetch SEC filings for {ticker}")
        else:
            filings = []
        
        # 2. Fetch Finnhub metrics (if use_rag is True)
        if use_rag:
            finnhub_metrics = self.fetch_metrics(ticker)
        else:
            finnhub_metrics = None
            self._log("Skipping Finnhub metrics (--no-rag flag)")
        
        # 3. Build RAG index with filings (if use_rag is True)
        if use_rag:
            self._log("Building RAG index...")
            
            # Clear previous documents
            self.rag.documents = []
            
            # Add filings
            self.rag.add_sec_filings(filings)
            
            # Add Finnhub metrics if available
            if finnhub_metrics:
                metrics_chunk = self.finnhub_client.format_metrics_for_rag(ticker)
                if metrics_chunk:
                    self.rag.add_financial_metrics(metrics_chunk)
            
            self.rag.build_index()
        else:
            self._log("RAG disabled (--no-rag flag) - using only LLM general knowledge")
            self._log("Skipping SEC filings and Finnhub metrics")
        
        # 4. Retrieve relevant context (prioritize SEC filings)
        if use_rag:
            self._log("Retrieving relevant context...")
            
            # Get SEC-specific context for risks and opportunities
            risk_context, risk_citations = self.rag.get_context_with_citations(
                f"What are the key risks and challenges for {company_name}?",
                top_k=5,
                source_types=['sec_filing']  # Prioritize SEC content
            )
        else:
            # Without RAG, no SEC filing context
            risk_context = ""
            risk_citations = []
        
        if use_rag:
            opp_context, opp_citations = self.rag.get_context_with_citations(
                f"What are the growth opportunities for {company_name}?",
                top_k=5,
                source_types=['sec_filing']  # Prioritize SEC content
            )
            
            # If SEC content is sparse, fall back to all sources
            if len(risk_context) < 100:
                risk_context, risk_citations = self.rag.get_context_with_citations(
                    f"What are the key risks?", top_k=3
                )
            if len(opp_context) < 100:
                opp_context, opp_citations = self.rag.get_context_with_citations(
                    f"What are growth opportunities?", top_k=3
                )
        else:
            opp_context = ""
            opp_citations = []
        
        # 5. Extract structured information
        filing_content = ""
        if filings and len(filings) > 0:
            filing_content = filings[0].get('content', '')[:50000]
        
        risks = self._extract_risks_from_content(risk_context, risk_citations)
        opportunities = self._extract_opportunities_from_content(opp_context, opp_citations)
        
        # 6. Build metrics list
        metrics_list = []
        metrics_interpretation = ""
        if finnhub_metrics:
            if finnhub_metrics.market_cap:
                # Finnhub returns market cap in millions
                if finnhub_metrics.market_cap >= 1e6:  # >= $1T
                    cap_str = f"${finnhub_metrics.market_cap / 1e6:.2f}T"
                elif finnhub_metrics.market_cap >= 1e3:  # >= $1B
                    cap_str = f"${finnhub_metrics.market_cap / 1e3:.2f}B"
                else:
                    cap_str = f"${finnhub_metrics.market_cap:.2f}M"
                metrics_list.append(FinancialMetric("Market Cap", cap_str))
            
            if finnhub_metrics.pe_ratio:
                metrics_list.append(FinancialMetric("P/E Ratio", f"{finnhub_metrics.pe_ratio:.1f}"))
            
            if finnhub_metrics.eps_ttm:
                metrics_list.append(FinancialMetric("EPS (TTM)", f"${finnhub_metrics.eps_ttm:.2f}"))
            
            if finnhub_metrics.revenue_growth_yoy is not None:
                sign = "+" if finnhub_metrics.revenue_growth_yoy >= 0 else ""
                metrics_list.append(FinancialMetric(
                    "Revenue Growth (YoY)", 
                    f"{sign}{finnhub_metrics.revenue_growth_yoy:.1f}%"
                ))
            
            if finnhub_metrics.debt_to_equity:
                metrics_list.append(FinancialMetric("Debt-to-Equity", f"{finnhub_metrics.debt_to_equity:.2f}"))
            
            metrics_interpretation = finnhub_metrics.get_student_interpretation()
        
        # 8. Generate company summary
        # First try to extract from SEC filing
        company_summary = self._get_company_summary(ticker, company_name, filing_content)
        
        # If model is enabled, use it to enhance/refine the summary
        if use_model:
            self._load_model()
            
            # Debug: Show what model is being used
            model_type = type(self.model).__name__
            print(f"[FINBRIEF] Using model type: {model_type}")
            if model_type == "DukeGatewayModel":
                print(f"[FINBRIEF] DUKE GATEWAY IS ACTIVE - Calling GPT 4.1 via Duke API")
            else:
                print(f"[FINBRIEF] Using local model: {model_type} (NOT Duke Gateway)")
            
            # If we have a good SEC summary, ask model to make it more beginner-friendly
            if use_rag and company_summary and "publicly traded company" not in company_summary:
                # Get comprehensive RAG context for company summary
                summary_context, summary_citations = self.rag.get_context_with_citations(
                    f"What does {company_name} do? What is their business model and main products or services?",
                    top_k=5,
                    source_types=['sec_filing']
                )
                
                # Combine extracted summary with RAG context for richer information
                combined_context = f"{company_summary[:500]}\n\nAdditional context from SEC filings:\n{summary_context}"
                
                # Load prompt from config
                enhancement_prompt = get_prompt('company_summary_enhancement', 
                                               company_name=company_name, 
                                               ticker=ticker)
                
                print(f"[FINBRIEF] Calling model.analyze_with_context() for company summary enhancement...")
                print(f"[FINBRIEF] RAG context length: {len(combined_context)} characters")
                enhanced = self.model.analyze_with_context(
                    combined_context,
                    enhancement_prompt,
                    for_beginners=True
                )
                if self._is_valid_summary(enhanced):
                    company_summary = enhanced
            else:
                # Get rich context about the company
                if use_rag:
                    company_context, company_citations = self.rag.get_context_with_citations(
                        get_prompt('rag_query_company_business', company_name=company_name),
                        top_k=8,
                        source_types=['sec_filing']
                    )
                else:
                    # Without RAG, minimal context
                    company_context = f"Company: {company_name} ({ticker})\n\n"
                    company_citations = []
                
                # Add financial metrics context
                clean_context = company_context
                if finnhub_metrics:
                    metrics_text = f"\n\nFinancial Metrics:\n"
                    metrics_text += f"Company: {company_name} ({ticker})\n"
                    if finnhub_metrics.market_cap:
                        metrics_text += f"Market Cap: ${finnhub_metrics.market_cap/1e9:.1f}B\n"
                    if finnhub_metrics.pe_ratio:
                        metrics_text += f"P/E Ratio: {finnhub_metrics.pe_ratio:.1f}\n"
                    if finnhub_metrics.revenue_growth_yoy is not None:
                        sign = "+" if finnhub_metrics.revenue_growth_yoy >= 0 else ""
                        metrics_text += f"Revenue Growth: {sign}{finnhub_metrics.revenue_growth_yoy:.1f}%\n"
                    clean_context += metrics_text
                
                # Add note if RAG is disabled
                if not use_rag:
                    clean_context += "\n\nNOTE: You do NOT have access to SEC filings. Use only your general knowledge about this company."
                
                # Load prompt from config
                summary_prompt = get_prompt('company_summary_generation',
                                           company_name=company_name,
                                           ticker=ticker)
                
                print(f"\n[RICH MODE] Generating summary...")
                print(f"[FINBRIEF] Calling model.analyze_with_context() to generate company summary...")
                print(f"[FINBRIEF] Context length: {len(clean_context)} characters")
                model_summary = self.model.analyze_with_context(
                    clean_context,
                    summary_prompt,
                    for_beginners=True
                )
                if self._is_valid_summary(model_summary):
                    company_summary = model_summary
                    # Truncate to standard mode limit
                    if len(company_summary) > 500:
                        print(f"[STANDARD MODE] Truncating output from {len(company_summary)} to 500 chars")
                        company_summary = company_summary[:500] + "..."
        
        # 9. Determine difficulty
        difficulty, difficulty_reason = self._determine_difficulty(filing_content, ticker)
        
        # 10. Generate investor takeaway
        default_takeaway = (
            f"{company_name} has both opportunities and risks that students should understand. "
            "Strong companies don't always make the best investments at any price. "
            "Consider the balance between growth potential and current valuation."
        )
        
        if use_model and risks and opportunities:
            # Ensure model is loaded
            if self.model is None:
                self._load_model()
            
            # Build comprehensive context for investor takeaway
            takeaway_context = f"""Company: {company_name} ({ticker})

Key Risks Identified:
{chr(10).join([f"- {r.description} (Severity: {r.severity.value})" for r in risks[:3]])}

Key Opportunities Identified:
{chr(10).join([f"- {o.description}" for o in opportunities[:3]])}

Financial Context:
"""
            if finnhub_metrics:
                if finnhub_metrics.market_cap:
                    takeaway_context += f"Market Cap: ${finnhub_metrics.market_cap/1e9:.1f}B\n"
                if finnhub_metrics.pe_ratio:
                    takeaway_context += f"P/E Ratio: {finnhub_metrics.pe_ratio:.1f}\n"
                if finnhub_metrics.revenue_growth_yoy is not None:
                    sign = "+" if finnhub_metrics.revenue_growth_yoy >= 0 else ""
                    takeaway_context += f"Revenue Growth: {sign}{finnhub_metrics.revenue_growth_yoy:.1f}%\n"
            
            # Load prompt from config
            takeaway_prompt = get_prompt('investor_takeaway',
                                        company_name=company_name,
                                        ticker=ticker)
            
            print(f"\n[STANDARD MODE] Generating investor takeaway...")
            print(f"[STANDARD MODE] Output will be truncated to ~400 chars")
            print(f"[STANDARD MODE] Use --rich flag for full analysis\n")
            print(f"[FINBRIEF] Calling model.analyze_with_context() for investor takeaway...")
            print(f"[FINBRIEF] Takeaway context includes {len(risks)} risks and {len(opportunities)} opportunities")
            model_takeaway = self.model.analyze_with_context(
                takeaway_context,
                takeaway_prompt,
                for_beginners=True
            )
            # Validate model output
            if self._is_valid_summary(model_takeaway):
                investor_takeaway = model_takeaway
                # Truncate to standard mode limit
                if len(investor_takeaway) > 400:
                    print(f"[STANDARD MODE] Truncating output from {len(investor_takeaway)} to 400 chars")
                    investor_takeaway = investor_takeaway[:400] + "..."
            else:
                investor_takeaway = default_takeaway
        else:
            investor_takeaway = default_takeaway
        
        # 11. Generate educational summary
        educational_summary = (
            f"This brief analyzes {company_name} using SEC filings and financial metrics. "
            f"Students should focus on understanding the business model, key risks, and "
            f"how financial metrics help evaluate companies. Remember: this is for learning, "
            f"not investment advice."
        )
        
        # 12. Build sources list
        sources = []
        if filings and len(filings) > 0:
            sources.append(Citation("SEC Filing", f"{filing_type} ({filings[0].get('filing_date', 'Recent')})"))
        if finnhub_metrics:
            sources.append(Citation("Finnhub", "Real-time financial metrics"))
        
        # 13. Compute confidence score
        all_citations = risk_citations + opp_citations
        term_explanations = []  # No longer using definitions corpus
        confidence_score, confidence_breakdown = self.confidence_estimator.estimate(
            citations=all_citations,
            has_metrics=finnhub_metrics is not None,
            content_length=len(company_summary) + len(investor_takeaway),
            analysis_sections=len([s for s in [risks, opportunities, metrics_list, term_explanations] if s])
        )
        self._log(f"Confidence score: {confidence_score:.2f}")
        
        # 14. Create the brief
        brief = EducationalBrief(
            ticker=ticker,
            company_name=company_name,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            company_summary=company_summary[:500] if company_summary else f"Analysis of {company_name}",
            summary_citations=[f"({filing_type} Filing)"],
            metrics=metrics_list,
            metrics_interpretation=metrics_interpretation,
            opportunities=opportunities,
            risks=risks,
            investor_takeaway=investor_takeaway[:400] if investor_takeaway else "",
            terms_explained=term_explanations,
            difficulty=difficulty,
            difficulty_reason=difficulty_reason,
            educational_summary=educational_summary,
            sources=sources,
            confidence_score=confidence_score
        )
        
        self._log("Brief generated successfully!")
        
        # Final debug summary
        if self.model:
            model_type = type(self.model).__name__
            print(f"\n{'='*80}")
            print(f"[FINAL SUMMARY] Model used: {model_type}")
            if model_type == "DukeGatewayModel":
                print(f"[FINAL SUMMARY] ✅ DUKE GATEWAY WAS USED - GPT 4.1 via Duke API")
            else:
                print(f"[FINAL SUMMARY] ⚠️  LOCAL MODEL WAS USED - {model_type} (NOT Duke Gateway)")
            print(f"{'='*80}\n")
        
        return brief
    
    def generate_rich_analysis(
        self, 
        ticker: str, 
        filing_type: str = "10-K",
        use_rag: bool = True
    ) -> EducationalBrief:
        """
        Generate comprehensive rich analysis with LLM-driven research.
        
        This mode:
        - Fetches comprehensive SEC and financial data (if use_rag=True)
        - Retrieves maximum RAG context (15 chunks) (if use_rag=True)
        - Makes a single comprehensive LLM call
        - Produces 1000-2000 word analysis
        - NO truncation or artificial limits
        
        Args:
            ticker: Stock ticker symbol
            filing_type: Type of SEC filing to analyze
            use_rag: Whether to use RAG for SEC filing retrieval (default: True)
            
        Returns:
            EducationalBrief with rich analysis content
        """
        ticker = ticker.upper()
        company_name = self._get_company_name(ticker)
        
        self._log(f"Generating RICH analysis for {company_name} ({ticker})")
        
        # Ensure model is loaded
        if self.model is None:
            self._load_model()
        
        # 1. Fetch SEC filings (if use_rag is True)
        if use_rag:
            filings = self.fetch_sec_filings(ticker, filing_type=filing_type, limit=1)
            if not filings:
                raise ValueError(f"Could not fetch SEC filings for {ticker}")
        else:
            filings = []
        
        # 2. Fetch Finnhub metrics (if use_rag is True)
        if use_rag:
            finnhub_metrics = self.fetch_metrics(ticker)
        else:
            finnhub_metrics = None
        
        # 3. Build comprehensive RAG index (if use_rag is True)
        if use_rag:
            self._log("Building comprehensive RAG index...")
            
            # Clear previous documents
            self.rag.documents = []
            
            # Add filings
            self.rag.add_sec_filings(filings)
            
            # Add Finnhub metrics if available
            if finnhub_metrics:
                metrics_chunk = self.finnhub_client.format_metrics_for_rag(ticker)
                if metrics_chunk:
                    self.rag.add_financial_metrics(metrics_chunk)
            
            # Rebuild index
            self.rag.build_index()
            
            # 4. Retrieve comprehensive context (15 chunks for rich mode)
            self._log("Retrieving comprehensive context (15 chunks)...")
            
            comprehensive_query = (
                f"Extract and synthesize comprehensive information about {company_name} from the SEC filing content provided. "
                f"Focus on: business description, financial metrics, key risks, growth opportunities, "
                f"competitive position, and market dynamics. Use specific details from the SEC filing sections in the context."
            )
            
            comprehensive_context, comprehensive_citations = self.rag.get_context_with_citations(
                comprehensive_query,
                top_k=15,  # Maximum context for rich mode
                source_types=['sec_filing', 'financial_metrics']
            )
        else:
            self._log("RAG disabled (--no-rag flag) - using only LLM general knowledge")
            # Without RAG, provide minimal context
            comprehensive_context = f"Company: {company_name} ({ticker})\n\n"
            comprehensive_context += "NOTE: You do NOT have access to SEC filings or financial metrics APIs. Use only your general knowledge about this company.\n\n"
            comprehensive_citations = []
        
        # 5. Add financial metrics to context
        if finnhub_metrics:
            metrics_text = f"\n\n{'='*60}\nFINANCIAL METRICS\n{'='*60}\n"
            metrics_text += f"Company: {company_name} ({ticker})\n"
            if finnhub_metrics.market_cap:
                metrics_text += f"Market Cap: ${finnhub_metrics.market_cap/1e9:.1f}B\n"
            if finnhub_metrics.pe_ratio:
                metrics_text += f"P/E Ratio: {finnhub_metrics.pe_ratio:.1f}\n"
            if finnhub_metrics.eps_ttm:
                metrics_text += f"EPS (TTM): ${finnhub_metrics.eps_ttm:.2f}\n"
            if finnhub_metrics.revenue_growth_yoy is not None:
                sign = "+" if finnhub_metrics.revenue_growth_yoy >= 0 else ""
                metrics_text += f"Revenue Growth (YoY): {sign}{finnhub_metrics.revenue_growth_yoy:.1f}%\n"
            if finnhub_metrics.debt_to_equity is not None:
                metrics_text += f"Debt-to-Equity Ratio: {finnhub_metrics.debt_to_equity:.2f}\n"
            comprehensive_context += metrics_text
        
        # 6. Load rich analysis prompt
        rich_prompt = get_prompt('rich_analysis_prompt', 
                                company_name=company_name, 
                                ticker=ticker)
        
        print(f"\n[RICH MODE] Generating comprehensive analysis...")
        print(f"[RICH MODE] Context length: {len(comprehensive_context)} characters")
        print(f"[RICH MODE] Citations: {len(comprehensive_citations)} sources")
        print(f"[RICH MODE] Model: {type(self.model).__name__}")
        print(f"[RICH MODE] Expected output: 1000-2000 words")
        print(f"[RICH MODE] NO TRUNCATION - Full LLM output will be preserved\n")
        
        # 7. Generate comprehensive analysis
        print(f"[FINBRIEF] Calling model.analyze_with_context() for rich analysis...")
        rich_analysis_content = self.model.analyze_with_context(
            comprehensive_context,
            rich_prompt,
            for_beginners=True
        )
        
        print(f"\n[RICH MODE] Analysis complete!")
        print(f"[RICH MODE] Output length: {len(rich_analysis_content)} characters (~{len(rich_analysis_content.split())} words)")
        
        # 8. Build sources list
        sources = []
        if filings and len(filings) > 0:
            sources.append(Citation("SEC Filing", f"{filing_type} ({filings[0].get('filing_date', 'Recent')})"))
        if finnhub_metrics:
            sources.append(Citation("Finnhub", "Real-time financial metrics"))
        
        # 9. Create brief with rich content
        # For rich mode, we store the comprehensive analysis in company_summary
        # and use special formatting in format_brief
        
        # Collect unique SEC filing URLs by type (10-K, 10-Q)
        # Also get URLs from the filings list directly
        sec_filing_urls = {}
        
        # Get URLs from filings list
        if filings and len(filings) > 0:
            for filing in filings:
                filing_form = filing.get('form', filing_type)  # Use form from filing or default to filing_type
                url = filing.get('url', '')
                if url:
                    # Normalize form type (handle variations)
                    if '10-K' in filing_form or filing_form == '10-K':
                        sec_filing_urls['10-K'] = url
                    elif '10-Q' in filing_form or filing_form == '10-Q':
                        sec_filing_urls['10-Q'] = url
        
        # Also check citations for any additional URLs
        for c in comprehensive_citations:
            source_type = c.get('source_type', '')
            source_url = c.get('source_url', '')
            if source_type == 'sec_filing' and source_url:
                # Try to determine filing type from URL or use default
                # Most citations will be from the same filing type we fetched
                if '10-K' in source_url or filing_type == '10-K':
                    sec_filing_urls['10-K'] = source_url
                elif '10-Q' in source_url or filing_type == '10-Q':
                    sec_filing_urls['10-Q'] = source_url
        
        # Format sources as simple list: one link per filing type
        formatted_citations = []
        for form_type in ['10-K', '10-Q']:
            if form_type in sec_filing_urls:
                url = sec_filing_urls[form_type]
                formatted_citations.append(f"SEC {form_type} Filing\n    URL: {url}")
        
        brief = EducationalBrief(
            ticker=ticker,
            company_name=company_name,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            company_summary=rich_analysis_content,  # Full rich analysis
            summary_citations=formatted_citations,
            metrics=[],  # Metrics are included in rich analysis text
            metrics_interpretation="",
            opportunities=[],  # Included in rich analysis text
            risks=[],  # Included in rich analysis text
            investor_takeaway="",  # Included in rich analysis text
            terms_explained=[],  # Included in rich analysis text
            difficulty=DifficultyLevel.MODERATE,
            difficulty_reason="Comprehensive analysis suitable for serious learners",
            educational_summary=f"Comprehensive investment research report on {company_name} using SEC filings and real-time financial data.",
            sources=sources,
            confidence_score=0.95  # High confidence with comprehensive analysis
        )
        
        self._log("Rich analysis generated successfully!")
        
        # Final debug summary
        if self.model:
            model_type = type(self.model).__name__
            print(f"\n{'='*80}")
            print(f"[FINAL SUMMARY] Model used: {model_type}")
            if model_type == "DukeGatewayModel":
                print(f"[FINAL SUMMARY] ✅ DUKE GATEWAY WAS USED - GPT 4.1 via Duke API")
            else:
                print(f"[FINAL SUMMARY] ⚠️  LOCAL MODEL WAS USED - {model_type} (NOT Duke Gateway)")
            print(f"{'='*80}\n")
        
        return brief
    
    def format_brief(self, brief: EducationalBrief, format: str = "text", rich_mode: bool = False) -> str:
        """
        Format a brief for display.
        
        Args:
            brief: EducationalBrief object
            format: Output format ("text" or "markdown")
            rich_mode: Use rich formatter for comprehensive analysis
            
        Returns:
            Formatted string
        """
        if rich_mode:
            if format == "markdown" or format == "md":
                return self.rich_formatter.format_markdown(brief)
            else:
                return self.rich_formatter.format_text(brief)
        else:
            if format == "markdown" or format == "md":
                return self.formatter.format_markdown(brief)
            else:
                return self.formatter.format_text(brief)
    
    def quick_metrics(self, ticker: str) -> Optional[str]:
        """
        Quick lookup of financial metrics only (no SEC filings).
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Formatted metrics string or None
        """
        metrics = self.fetch_metrics(ticker)
        if metrics:
            return metrics.format_for_students() + "\n\n" + metrics.get_student_interpretation()
        return None


# CLI interface
def main():
    """Command-line interface for FinBrief."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FinBrief: Educational Financial Brief Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.finbrief AAPL              # Full analysis with RAG (SEC + Finnhub)
  python -m src.finbrief MSFT --format md  # Markdown output
  python -m src.finbrief IBM --no-rag -o ibm_no_rag.txt  # Without RAG (pure LLM)
"""
    )
    
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--format', '-f', choices=['text', 'markdown', 'md'], 
                       default='text', help='Output format')
    parser.add_argument('--filing', choices=['10-K', '10-Q'], default='10-K',
                       help='SEC filing type to analyze')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick mode: metrics only, no SEC analysis')
    parser.add_argument('--no-model', action='store_true',
                       help='Skip LLM generation (faster but less detailed)')
    parser.add_argument('--no-finnhub', action='store_true',
                       help='Skip Finnhub metrics')
    parser.add_argument('--output', '-o', type=str, help='Save to file')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--api-key', type=str, help='Finnhub API key')
    parser.add_argument('--duke-gateway', action='store_true',
                       help='Use Duke AI Gateway (requires LITELLM_TOKEN in .env)')
    parser.add_argument('--duke-model', type=str, default='GPT 4.1',
                       help='Duke Gateway model to use (default: GPT 4.1)')
    parser.add_argument('--no-duke-gateway', action='store_true',
                       help='Force use of local model (skip Duke Gateway)')
    parser.add_argument('--no-rag', action='store_true',
                       help='Disable RAG: skip both SEC filings and Finnhub metrics (use only LLM general knowledge)')
    
    args = parser.parse_args()
    
    # Rich mode is always enabled (default behavior)
    rich_mode = True
    
    # Determine Duke Gateway usage
    use_duke_gateway = None
    if args.duke_gateway:
        use_duke_gateway = True
    elif args.no_duke_gateway:
        use_duke_gateway = False
    # Otherwise auto-detect (None)
    
    # Initialize app
    app = FinBriefApp(
        finnhub_api_key=args.api_key,
        use_finnhub=not args.no_finnhub,
        use_duke_gateway=use_duke_gateway,
        duke_model=args.duke_model,
        verbose=not args.quiet
    )
    
    if args.quick:
        # Quick metrics only
        result = app.quick_metrics(args.ticker)
        if result:
            print(result)
        else:
            print(f"Could not fetch metrics for {args.ticker}")
        return
    
    # Generate full brief
    try:
        brief = app.generate_brief(
            args.ticker,
            filing_type=args.filing,
            use_model=not args.no_model,
            use_rag=not args.no_rag
        )
        
        format_type = 'markdown' if args.format in ['markdown', 'md'] else 'text'
        output = app.format_brief(brief, format=format_type, rich_mode=True)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Brief saved to {args.output}")
        else:
            print(output)
            
    except Exception as e:
        print(f"Error generating brief: {e}")
        raise


if __name__ == "__main__":
    main()

