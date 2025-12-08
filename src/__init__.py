"""
FinBrief: Educational Financial Brief Generator

A comprehensive tool for generating beginner-investor-oriented investment education using:
- SEC EDGAR filings (10-K, 10-Q, 8-K)
- Finnhub financial metrics
- RAG-based retrieval (MiniLM + FAISS)
- Duke AI Gateway (GPT 4.1) with local model fallback

Supports both CLI and programmatic usage.
"""

# Core components
from .sec_edgar_client import SECEdgarClient
from .rag_system import RAGSystem, DocumentChunk
from .model_handler import FinBriefModel
from .duke_gateway_model import DukeGatewayModel

# FinBrief components
from .finnhub_client import FinnhubClient, FinancialMetrics
from .investment_score import calculate_investment_score
from .educational_brief import (
    EducationalBrief, EducationalBriefFormatter,
    RiskItem, OpportunityItem, TermExplanation, FinancialMetric, Citation,
    DifficultyLevel, RiskSeverity
)
from .finbrief import FinBriefApp
from .confidence_head import ConfidenceHead, HeuristicConfidenceEstimator

__version__ = "2.0.0"
__all__ = [
    "FinBriefApp",
    "SECEdgarClient",
    "FinnhubClient",
    "FinancialMetrics",
    "calculate_investment_score",
    "RAGSystem",
    "DocumentChunk",
    "FinBriefModel",
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
    "ConfidenceHead",
    "HeuristicConfidenceEstimator",
]
