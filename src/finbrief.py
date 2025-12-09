"""
This file integrates SEC filings (edgartools), RAG retrieval (FAISS + MiniLM), sentiment analysis (DistilBERT), 
financial metrics (Finnhub), and LLM generation (Duke AI Gateway) to produce informative reports.

ATTRIBUTION: The majority of code in this file is AI-generated. I used assistance to help modularize code for prompt loading and also to help with error handling.
"""
import os
import re
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from dotenv import load_dotenv

from .rag_system import RAGSystem
from .sentiment_classifier import SentimentClassifier
from .clients.sec_edgar_client import SECEdgarClient as SECEdgarClientOld
from .clients.edgartools_client import EdgarToolsClient
from .clients.finnhub_client import FinnhubClient, FinancialMetrics
from .clients.duke_gateway_model import DukeGatewayModel
from .utils.prompt_loader import get_prompt, PromptLoader
from .utils.educational_brief import (
    EducationalBrief, EducationalBriefFormatter,
    RiskItem, OpportunityItem, TermExplanation, FinancialMetric, Citation,
    DifficultyLevel, RiskSeverity, SentimentAnalysis
)
from .utils.rich_formatter import RichAnalysisFormatter
from .utils.section_validator import SectionValidator
from .utils.balance_sheet_analyzer import BalanceSheetAnalyzer
from .utils.model_handler import FinBriefModel

load_dotenv(override=True)


class FinBriefApp:
    """
    Main FinBrief application for generating educational investment briefs.
    """

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
        """
        self.verbose = verbose
        self.use_finnhub = use_finnhub
        self.model_name = model_name
        
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
        
        self._log("Loading SEC EDGAR client (edgartools)...")
        name = os.getenv('SEC_EDGAR_NAME', 'Student')
        email = os.getenv('SEC_EDGAR_EMAIL', 'student@university.edu')

        try:
            self.sec_client = EdgarToolsClient(
                name=name,
                email=email
            )
            self._log("Using edgartools (better section extraction)")
        except Exception as e:
            self._log(f"edgartools unavailable, using legacy client: {e}")
            company_name = os.getenv('SEC_EDGAR_COMPANY_NAME', 'FinBriefApp')
            self.sec_client = SECEdgarClientOld(
                company_name=company_name,
                name=name,
                email=email
            )
        
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
        
        self._log("Loading RAG system (MiniLM embeddings)...")
        self.rag = RAGSystem()

        self.formatter = EducationalBriefFormatter()
        self.rich_formatter = RichAnalysisFormatter()

        self.model = None

        self.section_validator = SectionValidator(verbose=self.verbose)
        self.balance_sheet_analyzer = BalanceSheetAnalyzer(verbose=self.verbose)

        self.sentiment_classifier = None

        self._log("FinBrief initialized successfully!")
    
    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[FinBrief] {message}")
    
    def _load_sentiment_classifier(self):
        """Lazy load the sentiment classifier."""
        if self.sentiment_classifier is None:
            self._log("Loading sentiment classifier...")
            self.sentiment_classifier = SentimentClassifier()
            self._log("Sentiment classifier loaded")
    
    def _get_company_name(self, ticker: str) -> str:
        """
        Get company name from ticker.
        Tries Finnhub first, then edgartools, then uses generic fallback.
        """
        ticker = ticker.upper()

        if self.finnhub_client:
            try:
                profile = self.finnhub_client.get_company_profile(ticker)
                if profile and profile.get('name'):
                    return profile['name']
            except:
                pass

        try:
            from edgar import Company, set_identity
            import os
            set_identity(f"{os.getenv('SEC_EDGAR_NAME', 'Student')} {os.getenv('SEC_EDGAR_EMAIL', 'test@duke.edu')}")
            company = Company(ticker)
            if company and hasattr(company, 'name') and company.name:
                return company.name
        except:
            pass

        return f"{ticker} Corporation"
    
    def _load_model(self):
        """
        Lazy load the LLM model with priority:
        1. Duke AI Gateway (if enabled and token available)
        2. Local model (TinyLlama or specified)
        """
        if self.model is None:
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
            else:
                print(f"[MODEL LOADER] Duke Gateway not requested (use_duke_gateway=False)")

            print(f"[MODEL LOADER] Duke Gateway not used, falling back to local model")
            if self.model_name:
                model_name = self.model_name
            else:
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
        
        clean_content = re.sub(r'<[^>]+>', '', content)
        clean_content = re.sub(r'http[s]?://\S+', '', clean_content)
        clean_content = re.sub(r'\[Definition\]', '', clean_content)
        
        sentences = re.split(r'[.!?]\s+', clean_content)
        risk_keywords = ['risk', 'could adversely', 'may negatively', 'uncertainty', 
                        'challenge', 'threat', 'competition', 'regulatory', 'litigation',
                        'decline', 'loss', 'adverse', 'failure']
        
        filing_citation = "(SEC Filing)"
        for c in citations:
            if c.get('source_type') == 'sec_filing':
                filing_citation = f"(10-Q/10-K Filing)"
                break
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_lower = sentence.lower()
            
            key = sentence_lower[:50]
            if key in seen_descriptions:
                continue
            
            if 'example:' in sentence_lower or sentence.startswith('Example'):
                continue
                
            if any(kw in sentence_lower for kw in risk_keywords):
                if len(sentence) > 50 and len(sentence) < 250:
                    if sum(c.isalpha() for c in sentence) < len(sentence) * 0.5:
                        continue
                    
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
        
        clean_content = re.sub(r'<[^>]+>', '', content)
        clean_content = re.sub(r'http[s]?://\S+', '', clean_content)
        clean_content = re.sub(r'\[Definition\]', '', clean_content)
        
        sentences = re.split(r'[.!?]\s+', clean_content)
        opp_keywords = ['growth', 'expansion', 'opportunity', 'increase', 'innovation',
                       'new market', 'strong demand', 'competitive advantage', 'improved',
                       'successful', 'positive', 'momentum']
        
        filing_citation = "(SEC Filing)"
        for c in citations:
            if c.get('source_type') == 'sec_filing':
                filing_citation = f"(10-Q/10-K Filing)"
                break
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_lower = sentence.lower()
            
            key = sentence_lower[:50]
            if key in seen_descriptions:
                continue
            
            if 'example:' in sentence_lower or sentence.startswith('Example'):
                continue
            
            if any(kw in sentence_lower for kw in opp_keywords):
                if len(sentence) > 50 and len(sentence) < 250:
                    if sum(c.isalpha() for c in sentence) < len(sentence) * 0.5:
                        continue
                    
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
        
        company_first_word = company_name.split()[0] if company_name else ticker
        
        item1a_idx = filing_content.lower().find('item 1a')
        
        if item1a_idx > 0:
            search_start = min(2000, len(filing_content) // 4)
            search_content = filing_content[search_start:item1a_idx]
        else:
            search_start = min(2000, len(filing_content) // 4)
            search_content = filing_content[search_start:]
        
        patterns = [
            r'Company\s+Background\s+(The\s+Company\s+[^.]+\.[^.]+\.)',
            
            r'\bGeneral\s+(We\s+(?:operate|are|provide|design|manufacture|develop|sell)[^.]{30,500}\.)',
            
            r'\bGENERAL\s+([A-Z][A-Za-z]+\s+is\s+a\s+[^.]+\.[^.]+\.)',
            
            r'(The\s+Company\s+(?:designs?|manufactures?|provides?|develops?|is\s+a|operates?|offers?|pioneers?)[^.]+\.[^.]+\.)',
            
            r'(^|\n|\.\s+)(We\s+(?:are\s+a|operate|provide|design|manufacture|develop|sell)\s+[^.]+\.[^.]+\.)',
            
            rf'({re.escape(company_first_word)}\s+(?:is\s+(?:a|an|now\s+a|the)|pioneered?|provides?)[^.]+\.[^.]+\.)',
            
            r'(Our\s+(?:mission|business)\s+is\s+[^.]+\.[^.]+\.)',
            
            r'Business\s+Overview\s+([A-Z][^.]+\.[^.]+\.)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, search_content, re.DOTALL | re.IGNORECASE | re.MULTILINE)
            if match:
                raw_text = match.group(1) if match.lastindex >= 1 else match.group(0)
                raw_text = raw_text.strip()
                
                skip_words = ['forward-looking', 'may relate', 'uncertainties', 'risks', 
                             'commission file', 'registrant', 'pursuant to', 'table of contents',
                             'accelerated filer', 'emerging growth company', 'check mark',
                             'section 13', 'exchange act', '☒', '☐', '☑']
                if any(word in raw_text.lower() for word in skip_words):
                    continue
                
                if raw_text.count('☒') + raw_text.count('☐') + raw_text.count('☑') > 0:
                    continue
                
                summary = self._extract_clean_sentences(raw_text, max_sentences=2)
                if summary and len(summary) > 80:
                    return summary
        
        item_1_content = self.sec_client.extract_section(filing_content, 'item_1')
        if item_1_content:
            sentences = re.split(r'(?<=[.!?])\s+', item_1_content[:2000])
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 60 and len(sentence) < 400 and
                    sentence[0].isupper() and
                    not any(word in sentence.lower() for word in ['item ', 'table of', 'forward-looking', 'risks'])):
                    text = self._clean_extracted_text(sentence)
                    if len(text) > 60:
                        return text
        
        business_verbs = r'(?:designs?|manufactures?|provides?|develops?|operates?|offers?|sells?|pioneered?|creates?|are\s+a)'
        sentences = re.split(r'(?<=[.!?])\s+', search_content[:15000])
        
        for sentence in sentences:
            sentence = sentence.strip()
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
        
        text = text.replace('\n', ' ').strip()
        
        if len(text) <= max_length:
            return text
        
        truncated = text[:max_length]
        
        sentence_endings = ['. ', '! ', '? ']
        last_sentence_end = -1
        
        search_start = max(0, max_length - 100)
        for i in range(len(truncated) - 1, search_start, -1):
            if i < len(truncated) - 1:
                if truncated[i] in ['.', '!', '?']:
                    if i + 1 < len(truncated) and truncated[i + 1] in [' ', '\t']:
                        last_sentence_end = i + 1
                        break
        
        if last_sentence_end > max_length * 0.5:
            excerpt = truncated[:last_sentence_end].strip()
        else:
            last_space = truncated.rfind(' ', 0, max_length)
            if last_space > max_length * 0.7:
                excerpt = truncated[:last_space].strip()
            else:
                excerpt = truncated.strip()
        
        if len(text) > max_length:
            excerpt += "..."
        
        return excerpt
    
    def _extract_clean_sentences(self, text: str, max_sentences: int = 3) -> str:
        """Extract clean, complete sentences from text."""
        import re
        
        text = self._clean_extracted_text(text)
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        clean_sentences = []
        for s in sentences[:max_sentences]:
            s = s.strip()
            if len(s) > 40 and s[0].isupper() and '.' in s:
                if not re.match(r'^Item\s+\d', s, re.IGNORECASE):
                    clean_sentences.append(s)
        
        return ' '.join(clean_sentences)
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text from SEC filing."""
        import re
        text = re.sub(r'\b[a-z]{2,10}:[A-Za-z0-9]+\b', '', text)
        text = re.sub(r'®|™|©', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _clean_sec_text(self, text: str, max_sentences: int = 4) -> str:
        """
        Aggressively clean SEC filing text to remove XBRL/XML artifacts.
        """
        if not text:
            return ""
        
        clean = re.sub(r'<[^>]+>', ' ', text)
        
        clean = re.sub(r'\b[a-z]{2,10}:[A-Za-z0-9]+\b', '', clean)
        clean = re.sub(r'\biso\d+:[A-Z]+\b', '', clean)
        
        clean = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', clean)
        clean = re.sub(r'\b\d{10,}\b', '', clean)
        clean = re.sub(r'\b0{6,}\d+\b', '', clean)
        
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
        
        clean = re.sub(r'\|+', ' ', clean)
        clean = re.sub(r'-{3,}', ' ', clean)
        clean = re.sub(r'_{3,}', ' ', clean)
        clean = re.sub(r'☒|☐', '', clean)
        
        clean = re.sub(r'[\r\n\t]+', ' ', clean)
        clean = re.sub(r'\s{2,}', ' ', clean)
        clean = clean.strip()
        
        sentences = re.split(r'(?<=[.!?])\s+', clean)
        good_sentences = []
        
        for s in sentences:
            s = s.strip()
            
            if len(s) < 40 or len(s) > 400:
                continue
            
            if not s or not s[0].isalpha():
                continue
            
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in s) / len(s)
            if alpha_ratio < 0.75:
                continue
            
            if re.search(r'[a-z]+:[A-Z]', s) or 'fasb.org' in s.lower():
                continue
            
            if s.lower().startswith(('item ', 'part ', 'table of')):
                continue
            
            good_sentences.append(s)
            if len(good_sentences) >= max_sentences:
                break
        
        return ' '.join(good_sentences)
    
    def _is_valid_summary(self, text: str) -> bool:
        """
        Check if generated text is a valid summary (not garbage).
        """
        if not text or len(text) < 50:
            return False
        
        bad_phrases = [
            "You are an educational",
            "Your goal is to explain",
            "Context from SEC filings",
            "Commission File Number",
            "Exact name of Registrant",
            "transition period from",
            "---|---|---",
            "☒", "☐",
            "What does this company do?",
            "Company Summary:",
        ]
        
        for phrase in bad_phrases:
            if phrase in text:
                return False
        
        words = text.lower().split()
        if len(words) > 10:
            from collections import Counter
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            if trigram_counts and trigram_counts.most_common(1)[0][1] > 3:
                return False
        
        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count < len(text) * 0.4:
            return False
        
        sentences = text.split('.')
        if len(sentences) < 2:
            return False
        
        return True
    
    def _extract_clean_summary(self, content: str, company_name: str) -> str:
        """Extract a clean company summary from filing content."""
        clean = re.sub(r'<[^>]+>', ' ', content)
        clean = re.sub(r'http[s]?://\S+', '', clean)
        clean = re.sub(r'\d{10}', '', clean)
        clean = re.sub(r'\|+', ' ', clean)
        clean = re.sub(r'-{3,}', ' ', clean)
        clean = re.sub(r'_{3,}', ' ', clean)
        clean = re.sub(r'[\r\n\t]+', ' ', clean)
        clean = re.sub(r'\s{2,}', ' ', clean)
        
        clean = re.sub(r'[a-z]+:[A-Za-z]+\d*', '', clean)
        clean = re.sub(r'iso\d+:[A-Z]+', '', clean)
        clean = re.sub(r'aapl:[A-Za-z]+', '', clean)
        clean = re.sub(r'\d{4}-\d{2}-\d{2}', '', clean)
        clean = re.sub(r'\b[A-Z]{2,}[a-z]+[A-Z][a-z]+\b', '', clean)
        
        skip_patterns = [
            r'united states securities',
            r'commission file number',
            r'exact name of registrant',
            r'state or other jurisdiction',
            r'i\.r\.s\.',
            r'false\d{4}',
            r'washington, d\.c\.',
            r'p\d+y',
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
        
        sentences = re.split(r'[.!?]\s+', clean)
        good_sentences = []
        
        for s in sentences:
            s = s.strip()
            s_lower = s.lower()
            
            if len(s) < 50 or len(s) > 300:
                continue
            
            if not s[0].isalpha():
                continue
            
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in s) / len(s)
            if alpha_ratio < 0.7:
                continue
            
            if 'fasb.org' in s_lower or 'gaap' in s_lower:
                continue
            
            if any(re.search(pattern, s_lower) for pattern in skip_patterns):
                continue
            
            good_sentences.append(s)
            if len(good_sentences) >= 3:
                break
        
        if good_sentences:
            return '. '.join(good_sentences) + '.'
        
        return ""
    
    def _determine_difficulty(self, content: str, ticker: str) -> Tuple[DifficultyLevel, str]:
        """Determine beginner difficulty level for understanding the company."""
        complex_terms = ['derivatives', 'hedging', 'amortization', 'goodwill impairment',
                        'deferred tax', 'contingent consideration', 'variable interest entity']
        
        content_lower = content.lower()
        complexity_count = sum(1 for term in complex_terms if term in content_lower)
        
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
        
        if use_rag:
            filings = self.fetch_sec_filings(ticker, filing_type=filing_type, limit=1)
            
            if not filings:
                raise ValueError(f"Could not fetch SEC filings for {ticker}")
        else:
            filings = []
        
        if use_rag:
            finnhub_metrics = self.fetch_metrics(ticker)
        else:
            finnhub_metrics = None
            self._log("Skipping Finnhub metrics (--no-rag flag)")
        
        if use_rag:
            self._log("Building RAG index...")
            
            self.rag.documents = []
            
            self.rag.add_sec_filings(filings)
            
            if finnhub_metrics:
                metrics_chunk = self.finnhub_client.format_metrics_for_rag(ticker)
                if metrics_chunk:
                    self.rag.add_financial_metrics(metrics_chunk)
            
            self.rag.build_index()
        else:
            self._log("RAG disabled (--no-rag flag) - using only LLM general knowledge")
            self._log("Skipping SEC filings and Finnhub metrics")
        
        if use_rag:
            self._log("Retrieving relevant context...")
            
            risk_context, risk_citations = self.rag.get_context_with_citations(
                f"What are the key risks and challenges for {company_name}?",
                top_k=None,
                source_types=['sec_filing'],
                purpose="RISKS ANALYSIS",
                min_coverage=0.4
            )
        else:
            risk_context = ""
            risk_citations = []
        
        if use_rag:
            opp_context, opp_citations = self.rag.get_context_with_citations(
                f"What are the growth opportunities for {company_name}?",
                top_k=None,
                source_types=['sec_filing'],
                purpose="OPPORTUNITIES ANALYSIS",
                min_coverage=0.4
            )
            
            if len(risk_context) < 100:
                risk_context, risk_citations = self.rag.get_context_with_citations(
                    f"What are the key risks?", top_k=3,
                    purpose="RISKS FALLBACK"
                )
            if len(opp_context) < 100:
                opp_context, opp_citations = self.rag.get_context_with_citations(
                    f"What are growth opportunities?", top_k=3,
                    purpose="OPPORTUNITIES FALLBACK"
                )
        else:
            opp_context = ""
            opp_citations = []
        
        filing_content = ""
        if filings and len(filings) > 0:
            filing_content = filings[0].get('content', '')[:50000]
        
        risks = self._extract_risks_from_content(risk_context, risk_citations)
        opportunities = self._extract_opportunities_from_content(opp_context, opp_citations)
        
        metrics_list = []
        metrics_interpretation = ""
        if finnhub_metrics:
            if finnhub_metrics.market_cap:
                if finnhub_metrics.market_cap >= 1e6:
                    cap_str = f"${finnhub_metrics.market_cap / 1e6:.2f}T"
                elif finnhub_metrics.market_cap >= 1e3:
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
        
        company_summary = self._get_company_summary(ticker, company_name, filing_content)
        
        if use_model:
            self._load_model()
            
            model_type = type(self.model).__name__
            print(f"[FINBRIEF] Using model type: {model_type}")
            if model_type == "DukeGatewayModel":
                print(f"[FINBRIEF] DUKE GATEWAY IS ACTIVE - Calling GPT 4.1 via Duke API")
            else:
                print(f"[FINBRIEF] Using local model: {model_type} (NOT Duke Gateway)")
            
            if use_rag and company_summary and "publicly traded company" not in company_summary:
                summary_context, summary_citations = self.rag.get_context_with_citations(
                    f"What does {company_name} do? What is their business model and main products or services?",
                    top_k=None,
                    source_types=['sec_filing'],
                    purpose="COMPANY SUMMARY",
                    min_coverage=0.3
                )
                
                combined_context = f"{company_summary[:500]}\n\nAdditional context from SEC filings:\n{summary_context}"
                
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
                if use_rag:
                    company_context, company_citations = self.rag.get_context_with_citations(
                        get_prompt('rag_query_company_business', company_name=company_name),
                        top_k=None,
                        source_types=['sec_filing'],
                        purpose="BUSINESS DESCRIPTION",
                        min_coverage=0.4
                    )
                else:
                    company_context = f"Company: {company_name} ({ticker})\n\n"
                    company_citations = []
                
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
                
                if not use_rag:
                    clean_context += "\n\nNOTE: You do NOT have access to SEC filings. Use only your general knowledge about this company."
                
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
                    if len(company_summary) > 500:
                        print(f"[STANDARD MODE] Truncating output from {len(company_summary)} to 500 chars")
                        company_summary = company_summary[:500] + "..."
        
        difficulty, difficulty_reason = self._determine_difficulty(filing_content, ticker)
        
        default_takeaway = (
            f"{company_name} has both opportunities and risks that students should understand. "
            "Strong companies don't always make the best investments at any price. "
            "Consider the balance between growth potential and current valuation."
        )
        
        if use_model and risks and opportunities:
            if self.model is None:
                self._load_model()
            
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
            if self._is_valid_summary(model_takeaway):
                investor_takeaway = model_takeaway
                if len(investor_takeaway) > 400:
                    print(f"[STANDARD MODE] Truncating output from {len(investor_takeaway)} to 400 chars")
                    investor_takeaway = investor_takeaway[:400] + "..."
            else:
                investor_takeaway = default_takeaway
        else:
            investor_takeaway = default_takeaway
        
        educational_summary = (
            f"This brief analyzes {company_name} using SEC filings and financial metrics. "
            f"Students should focus on understanding the business model, key risks, and "
            f"how financial metrics help evaluate companies. Remember: this is for learning, "
            f"not investment advice."
        )
        
        sources = []
        if filings and len(filings) > 0:
            sources.append(Citation("SEC Filing", f"{filing_type} ({filings[0].get('filing_date', 'Recent')})"))
        if finnhub_metrics:
            sources.append(Citation("Finnhub", "Real-time financial metrics"))
        
        term_explanations = []
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
            sources=sources
        )
        
        self._log("Brief generated successfully!")
        
        if self.model:
            model_type = type(self.model).__name__
            print(f"\n{'='*80}")
            print(f"[FINAL SUMMARY] Model used: {model_type}")
            if model_type == "DukeGatewayModel":
                actual_model = getattr(self.model, 'model_name', 'Unknown')
                print(f"[FINAL SUMMARY]  DUKE GATEWAY WAS USED - {actual_model} via Duke API")
            else:
                print(f"[FINAL SUMMARY]   LOCAL MODEL WAS USED - {model_type} (NOT Duke Gateway)")
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
        
        if self.model is None:
            self._load_model()
        
        if use_rag:
            filings = self.fetch_sec_filings(ticker, filing_type=filing_type, limit=1)
            if not filings:
                raise ValueError(f"Could not fetch SEC filings for {ticker}")
        else:
            filings = []

        if use_rag:
            finnhub_metrics = self.fetch_metrics(ticker)
        else:
            finnhub_metrics = None

        balance_sheet_context = ""
        section_status = {}
        sentiment_analysis_result = None

        if use_rag and filings:
            self._log("\n[PHASE 3A] Validating sections and analyzing balance sheet...")

            try:
                from edgar import Company, set_identity
                set_identity(f"{os.getenv('SEC_EDGAR_NAME', 'Student')} {os.getenv('SEC_EDGAR_EMAIL', 'test@duke.edu')}")
                company = Company(ticker)
                filing = company.get_filings(form=filing_type).latest()
                doc = filing.obj()

                validated_sections = {}

                for section_name in ['business', 'risk_factors', 'management_discussion']:
                    content, status = self.section_validator.get_section_with_fallback(
                        doc, section_name, company_name, verbose=self.verbose
                    )
                    validated_sections[section_name] = content
                    section_status[section_name] = status

                    if status == 'fallback_business' or status == 'fallback_full_filing':
                        self._log(f" [FALLBACK] {section_name}: Used {status} strategy")
                    elif status == 'failed':
                        self._log(f"  [WARNING] {section_name}: All extraction methods failed")

                if hasattr(doc, 'balance_sheet') and doc.balance_sheet:
                    self._log("Analyzing balance sheet...")
                    bs_analysis = self.balance_sheet_analyzer.analyze(doc.balance_sheet, company_name)

                    if bs_analysis['has_data']:
                        balance_sheet_context = "\n\n" + bs_analysis['formatted_context']
                        self._log(f" Balance sheet analysis complete")
                    else:
                        self._log("  Balance sheet data unavailable")
                else:
                    self._log("  Balance sheet not available in filing")

            except Exception as e:
                self._log(f"  Phase 3A processing failed: {e}")
                import traceback
                if self.verbose:
                    traceback.print_exc()

        if use_rag:
            self._log("Building comprehensive RAG index...")

            self.rag.documents = []

            if validated_sections:
                section_filings = []
                for section_name, content in validated_sections.items():
                    section_filings.append({
                        'content': content,
                        'section': section_name,
                        'ticker': ticker,
                        'filing_date': 'Latest',
                        'url': filings[0].get('url', '') if filings else ''
                    })
                self.rag.add_sec_filings(section_filings)
                self._log(f"Added {len(validated_sections)} validated sections to RAG index")
            elif filings:
                self.rag.add_sec_filings(filings)

            if finnhub_metrics:
                metrics_chunk = self.finnhub_client.format_metrics_for_rag(ticker)
                if metrics_chunk:
                    self.rag.add_financial_metrics(metrics_chunk)

            self.rag.build_index()
            
            if self.rag.documents:
                self._load_sentiment_classifier()
                self._log(f"Analyzing sentiment of {len(self.rag.documents)} chunks...")
                
                chunk_texts = [doc.text for doc in self.rag.documents]
                
                sentiment_predictions = self.sentiment_classifier.classify_batch(chunk_texts)
                
                pos_count = sum(1 for p in sentiment_predictions if p['label'] == 'positive')
                neu_count = sum(1 for p in sentiment_predictions if p['label'] == 'neutral')
                neg_count = sum(1 for p in sentiment_predictions if p['label'] == 'negative')
                total_count = len(sentiment_predictions)
                
                if total_count > 0:
                    positive_pct = (pos_count / total_count) * 100
                    neutral_pct = (neu_count / total_count) * 100
                    negative_pct = (neg_count / total_count) * 100
                    
                    if positive_pct > 50 and positive_pct > neutral_pct and positive_pct > negative_pct:
                        overall_tone = "Positive"
                    elif negative_pct > 50 and negative_pct > neutral_pct and negative_pct > positive_pct:
                        overall_tone = "Negative"
                    elif neutral_pct > 50 and neutral_pct > positive_pct and neutral_pct > negative_pct:
                        overall_tone = "Neutral"
                    else:
                        overall_tone = "Mixed"
                    
                    sentiment_analysis_result = SentimentAnalysis(
                        positive_pct=positive_pct,
                        neutral_pct=neutral_pct,
                        negative_pct=negative_pct,
                        total_chunks=total_count,
                        overall_tone=overall_tone
                    )
                    
                    self._log(f"Sentiment analysis complete: {overall_tone} "
                             f"(Pos: {positive_pct:.1f}%, Neu: {neutral_pct:.1f}%, Neg: {negative_pct:.1f}%)")
            else:
                self._log("No RAG chunks available for sentiment analysis")
            
            self._log("Retrieving comprehensive context (15 chunks)...")
            
            comprehensive_query = (
                f"Extract and synthesize comprehensive information about {company_name} from the SEC filing content provided. "
                f"Focus on: business description, qualitative risks from Risk Factors section, strategic opportunities, "
                f"competitive position, market dynamics, and financial metrics. "
                f"For risks, prioritize qualitative factors, regulatory challenges, market conditions, and strategic risks "
                f"from the Risk Factors section over balance sheet ratios. Use specific details from the SEC filing sections."
            )
            
            comprehensive_context, comprehensive_citations = self.rag.get_context_with_citations(
                comprehensive_query,
                top_k=None,
                source_types=['sec_filing', 'financial_metrics'],
                purpose="COMPREHENSIVE ANALYSIS",
                min_coverage=0.7,
                ensure_all_sections=True
            )
        else:
            self._log("RAG disabled (--no-rag flag) - using only LLM general knowledge")
            comprehensive_context = f"Company: {company_name} ({ticker})\n\n"
            comprehensive_context += "NOTE: You do NOT have access to SEC filings or financial metrics APIs. Use only your general knowledge about this company.\n\n"
            comprehensive_citations = []
        
        if use_rag and finnhub_metrics:
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

        if use_rag and balance_sheet_context:
            comprehensive_context += balance_sheet_context
            if self.verbose:
                print(f" [PHASE 3A] Added balance sheet analysis to context")

        if use_rag:
            rich_prompt = get_prompt('rich_analysis_prompt',
                                    company_name=company_name,
                                    ticker=ticker)
            system_instructions = get_prompt('system_instructions')
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            no_rag_prompts_file = os.path.join(project_root, 'config', 'prompts_no_rag.md')
            no_rag_loader = PromptLoader(prompts_file=no_rag_prompts_file)
            rich_prompt = no_rag_loader.get('rich_analysis_prompt',
                                           company_name=company_name,
                                           ticker=ticker)
            system_instructions = no_rag_loader.get('system_instructions')

        print(f"\n[RICH MODE] Generating comprehensive analysis...")
        print(f"[RICH MODE] Context length: {len(comprehensive_context)} characters")
        print(f"[RICH MODE] Citations: {len(comprehensive_citations)} sources")
        print(f"[RICH MODE] Model: {type(self.model).__name__}")
        print(f"[RICH MODE] Expected output: 1000-2000 words")
        print(f"[RICH MODE] NO TRUNCATION - Full LLM output will be preserved\n")

        print(f"[FINBRIEF] Calling model.analyze_with_context() for rich analysis...")
        rich_analysis_content = self.model.analyze_with_context(
            comprehensive_context,
            rich_prompt,
            for_beginners=True,
            system_instructions=system_instructions
        )
        
        print(f"\n[RICH MODE] Analysis complete!")
        print(f"[RICH MODE] Output length: {len(rich_analysis_content)} characters (~{len(rich_analysis_content.split())} words)")
        
        sources = []
        if filings and len(filings) > 0:
            sources.append(Citation("SEC Filing", f"{filing_type} ({filings[0].get('filing_date', 'Recent')})"))
        if finnhub_metrics:
            sources.append(Citation("Finnhub", "Real-time financial metrics"))
        
        
        sec_filing_urls = {}
        
        if filings and len(filings) > 0:
            for filing in filings:
                filing_form = filing.get('form', filing_type)
                url = filing.get('url', '')
                if url:
                    if '10-K' in filing_form or filing_form == '10-K':
                        sec_filing_urls['10-K'] = url
                    elif '10-Q' in filing_form or filing_form == '10-Q':
                        sec_filing_urls['10-Q'] = url
        
        for c in comprehensive_citations:
            source_type = c.get('source_type', '')
            source_url = c.get('source_url', '')
            if source_type == 'sec_filing' and source_url:
                if '10-K' in source_url or filing_type == '10-K':
                    sec_filing_urls['10-K'] = source_url
                elif '10-Q' in source_url or filing_type == '10-Q':
                    sec_filing_urls['10-Q'] = source_url
        
        formatted_citations = []
        for form_type in ['10-K', '10-Q']:
            if form_type in sec_filing_urls:
                url = sec_filing_urls[form_type]
                formatted_citations.append(f"SEC {form_type} Filing\n    URL: {url}")
        
        brief = EducationalBrief(
            ticker=ticker,
            company_name=company_name,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            company_summary=rich_analysis_content,
            summary_citations=formatted_citations,
            metrics=[],
            metrics_interpretation="",
            opportunities=[],
            risks=[],
            investor_takeaway="",
            terms_explained=[],
            difficulty=DifficultyLevel.MODERATE,
            difficulty_reason="Comprehensive analysis suitable for serious learners",
            educational_summary=f"Comprehensive investment research report on {company_name} using SEC filings and real-time financial data.",
            sources=sources,
            sentiment_analysis=sentiment_analysis_result
        )
        
        self._log("Rich analysis generated successfully!")
        
        if self.model:
            model_type = type(self.model).__name__
            print(f"\n{'='*80}")
            print(f"[FINAL SUMMARY] Model used: {model_type}")
            if model_type == "DukeGatewayModel":
                actual_model = getattr(self.model, 'model_name', 'Unknown')
                print(f"[FINAL SUMMARY]  DUKE GATEWAY WAS USED - {actual_model} via Duke API")
            else:
                print(f"[FINAL SUMMARY]   LOCAL MODEL WAS USED - {model_type} (NOT Duke Gateway)")
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


def _format_comparison(ticker: str, output_rag: str, output_no_rag: str) -> str:
    """
    Format side-by-side comparison of RAG vs No-RAG outputs.

    Args:
        ticker: Stock ticker
        output_rag: Report generated with RAG
        output_no_rag: Report generated without RAG

    Returns:
        Formatted comparison output
    """
    comparison = []
    comparison.append("="*100)
    comparison.append(f"RAG vs NO-RAG COMPARISON for {ticker}")
    comparison.append("="*100)
    comparison.append("")
    comparison.append("This comparison demonstrates the qualitative impact of Retrieval-Augmented Generation (RAG)")
    comparison.append("on report quality. The RAG report uses actual SEC filings and real-time financial data,")
    comparison.append("while the No-RAG report relies solely on the LLM's general knowledge.")
    comparison.append("")
    comparison.append("="*100)
    comparison.append("REPORT WITH RAG (SEC Filings + Finnhub)")
    comparison.append("="*100)
    comparison.append("")
    comparison.append(output_rag)
    comparison.append("")
    comparison.append("")
    comparison.append("="*100)
    comparison.append("REPORT WITHOUT RAG (LLM General Knowledge Only)")
    comparison.append("="*100)
    comparison.append("")
    comparison.append(output_no_rag)
    comparison.append("")
    comparison.append("")
    comparison.append("="*100)
    comparison.append("KEY DIFFERENCES TO NOTE:")
    comparison.append("="*100)
    comparison.append("")
    comparison.append("1. **Specificity**: RAG report includes specific numbers, dates, and facts from SEC filings")
    comparison.append("2. **Accuracy**: RAG report uses authoritative SEC data vs potentially outdated LLM knowledge")
    comparison.append("3. **Citations**: RAG report provides source attribution to SEC filings")
    comparison.append("4. **Depth**: RAG report includes company-specific risks and opportunities from actual filings")
    comparison.append("5. **Currency**: RAG report reflects latest financial data from Finnhub API")
    comparison.append("")
    comparison.append("Without RAG, the LLM may:")
    comparison.append("- Use outdated information from its training data")
    comparison.append("- Hallucinate facts or figures")
    comparison.append("- Provide generic analysis not tailored to recent company performance")
    comparison.append("- Miss recent strategic shifts or risk factors disclosed in latest filings")
    comparison.append("")

    return "\n".join(comparison)


def main():
    """Command-line interface for FinBrief."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FinBrief: Educational Financial Brief Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.finbrief AAPL                          # Full analysis with RAG (SEC + Finnhub)
  python -m src.finbrief MSFT --format md              # Markdown output
  python -m src.finbrief IBM --no-rag                  # Without RAG (pure LLM knowledge)
  python -m src.finbrief TSLA --compare -o comparison.txt  # RAG vs No-RAG side-by-side
  python -m src.finbrief NVDA --duke-model "GPT 4o"    # Use different Duke Gateway model

Available Duke Gateway models:
  GPT 4.1, GPT 4.1 Mini, GPT 4o, o4 Mini, Llama 3.3, Llama 4 Scout, Mistral on-site
  (Configure default in .env: DUKE_AI_MODEL="GPT 4.1")
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
    parser.add_argument('--duke-model', type=str,
                       help='Duke Gateway model to use (overrides DUKE_AI_MODEL in .env)')
    parser.add_argument('--no-duke-gateway', action='store_true',
                       help='Force use of local model (skip Duke Gateway)')
    parser.add_argument('--no-rag', action='store_true',
                       help='Disable RAG: skip both SEC filings and Finnhub metrics (use only LLM general knowledge)')
    parser.add_argument('--compare', action='store_true',
                       help='Generate both RAG and No-RAG reports side-by-side for comparison')

    args = parser.parse_args()
    
    rich_mode = True
    
    use_duke_gateway = None
    if args.duke_gateway:
        use_duke_gateway = True
    elif args.no_duke_gateway:
        use_duke_gateway = False

    app = FinBriefApp(
        finnhub_api_key=args.api_key,
        use_finnhub=not args.no_finnhub,
        use_duke_gateway=use_duke_gateway,
        duke_model=args.duke_model,
        verbose=not args.quiet
    )

    if args.quick:
        result = app.quick_metrics(args.ticker)
        if result:
            print(result)
        else:
            print(f"Could not fetch metrics for {args.ticker}")
        return

    if args.compare:
        print("="*80)
        print("RAG vs NO-RAG COMPARISON MODE")
        print(f"Generating two reports for {args.ticker} to demonstrate RAG impact")
        print("="*80)
        print()

        try:
            print("[1/2] Generating WITH RAG (SEC filings + Finnhub metrics)...")
            brief_with_rag = app.generate_brief(
                args.ticker,
                filing_type=args.filing,
                use_model=True,
                use_rag=True
            )

            print("[2/2] Generating WITHOUT RAG (LLM general knowledge only)...")
            brief_no_rag = app.generate_brief(
                args.ticker,
                filing_type=args.filing,
                use_model=True,
                use_rag=False
            )

            format_type = 'markdown' if args.format in ['markdown', 'md'] else 'text'
            output_rag = app.format_brief(brief_with_rag, format=format_type, rich_mode=True)
            output_no_rag = app.format_brief(brief_no_rag, format=format_type, rich_mode=True)

            comparison_output = _format_comparison(args.ticker, output_rag, output_no_rag)

            if args.output:
                with open(args.output, 'w') as f:
                    f.write(comparison_output)
                print(f"Comparison saved to {args.output}")
            else:
                print(comparison_output)

        except Exception as e:
            print(f"Error generating comparison: {e}")
            raise
        return

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

