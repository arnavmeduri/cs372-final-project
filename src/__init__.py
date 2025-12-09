"""
FinBrief: Educational Financial Brief Generator

A comprehensive tool for generating beginner-investor-oriented investment education using
SEC EDGAR filings (10-K, 10-Q), Finnhub financial metrics, RAG-based retrieval (MiniLM + FAISS)
sentiment analysis (DistilBERT), and LLM generation (Duke AI Gateway).
"""

# Core ML components
from .rag_system import RAGSystem, DocumentChunk
from .sentiment_classifier import SentimentClassifier
from .finbrief import FinBriefApp

# API clients
from .clients import (
    SECEdgarClient,
    EdgarToolsClient,
    FinnhubClient,
    FinancialMetrics,
    DukeGatewayModel
)

# Utilities
from .utils import (
    EducationalBrief,
    EducationalBriefFormatter,
    RiskItem,
    OpportunityItem,
    TermExplanation,
    FinancialMetric,
    Citation,
    DifficultyLevel,
    RiskSeverity,
    SentimentAnalysis,
    RichAnalysisFormatter,
    PromptLoader,
    get_prompt,
    SectionValidator,
    BalanceSheetAnalyzer,
    FinBriefModel
)

__version__ = "2.0.0"
__all__ = [
    "FinBriefApp",
    "RAGSystem",
    "DocumentChunk",
    "SentimentClassifier",
    "SECEdgarClient",
    "EdgarToolsClient",
    "FinnhubClient",
    "FinancialMetrics",
    "DukeGatewayModel",
    "EducationalBrief",
    "EducationalBriefFormatter",
    "RiskItem",
    "OpportunityItem",
    "TermExplanation",
    "FinancialMetric",
    "Citation",
    "DifficultyLevel",
    "RiskSeverity",
    "SentimentAnalysis",
    "RichAnalysisFormatter",
    "PromptLoader",
    "get_prompt",
    "SectionValidator",
    "BalanceSheetAnalyzer",
    "FinBriefModel",
]
