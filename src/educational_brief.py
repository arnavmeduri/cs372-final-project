"""
FinBrief: Educational financial brief output format.
Generates student-oriented, beginner-friendly investment analysis.

Based on design.md output format:
1. Company Summary (Beginner-Friendly)
2. Key Financial Metrics for Students
3. Opportunities (Based on SEC filings)
4. Risks Students Should Understand
5. What This Means for a New Investor
6. Key Terms Explained
7. Beginner Difficulty Score
8. Educational Summary (Not Investment Advice)
9. Retrieval Sources
10. Model-Estimated Confidence (optional)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import json


class DifficultyLevel(Enum):
    """Beginner difficulty levels for understanding the company."""
    EASY = "Easy"
    MODERATE = "Moderate"
    DIFFICULT = "Difficult"


class RiskSeverity(Enum):
    """Risk severity levels."""
    HIGH = "High"
    MEDIUM = "Medium"
    MEDIUM_HIGH = "Medium-High"
    LOW = "Low"


@dataclass
class RiskItem:
    """A risk that students should understand."""
    description: str
    severity: RiskSeverity
    citation: str  # e.g., "(10-K Item 1A)"
    explanation: str = ""  # Optional beginner-friendly explanation


@dataclass
class OpportunityItem:
    """A growth opportunity from SEC filings."""
    description: str
    citation: str  # e.g., "(10-K Item 7)"
    category: str = ""  # e.g., "Services Growth", "Market Expansion"


@dataclass
class TermExplanation:
    """A financial term explanation for beginners."""
    term: str
    definition: str
    source: str  # "Investor.gov" or "Investopedia"


@dataclass
class FinancialMetric:
    """A financial metric with student interpretation."""
    name: str
    value: str
    interpretation: str = ""


@dataclass
class Citation:
    """A source citation."""
    source_type: str  # 'SEC Filing', 'Definitions', 'Finnhub'
    details: str  # e.g., "10-K (Sections 1, 1A, 7)"


@dataclass
class SentimentAnalysis:
    """Sentiment analysis of SEC filing chunks."""
    positive_pct: float  # Percentage of positive chunks
    neutral_pct: float   # Percentage of neutral chunks
    negative_pct: float  # Percentage of negative chunks
    total_chunks: int    # Number of chunks analyzed
    overall_tone: str    # "Positive", "Neutral", "Negative", or "Mixed"


@dataclass
class EducationalBrief:
    """
    Complete FinBrief educational investment brief.
    Structured for beginner investors learning about companies.
    """
    # Header
    ticker: str
    company_name: str
    analysis_date: str
    
    # 1. Company Summary
    company_summary: str
    summary_citations: List[str] = field(default_factory=list)
    
    # 2. Key Financial Metrics
    metrics: List[FinancialMetric] = field(default_factory=list)
    metrics_interpretation: str = ""
    
    # 3. Opportunities
    opportunities: List[OpportunityItem] = field(default_factory=list)
    
    # 4. Risks
    risks: List[RiskItem] = field(default_factory=list)
    
    # 5. What This Means for a New Investor
    investor_takeaway: str = ""
    
    # 6. Key Terms Explained
    terms_explained: List[TermExplanation] = field(default_factory=list)
    
    # 7. Beginner Difficulty Score
    difficulty: DifficultyLevel = DifficultyLevel.MODERATE
    difficulty_reason: str = ""
    
    # 8. Educational Summary
    educational_summary: str = ""
    
    # 9. Retrieval Sources
    sources: List[Citation] = field(default_factory=list)
    
    # 10. Model Confidence (optional)
    confidence_score: Optional[float] = None
    
    # 11. Sentiment Analysis (optional)
    sentiment_analysis: Optional[SentimentAnalysis] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'analysis_date': self.analysis_date,
            'company_summary': self.company_summary,
            'summary_citations': self.summary_citations,
            'metrics': [{'name': m.name, 'value': m.value, 'interpretation': m.interpretation} 
                       for m in self.metrics],
            'metrics_interpretation': self.metrics_interpretation,
            'opportunities': [{'description': o.description, 'citation': o.citation, 'category': o.category}
                            for o in self.opportunities],
            'risks': [{'description': r.description, 'severity': r.severity.value, 
                      'citation': r.citation, 'explanation': r.explanation}
                     for r in self.risks],
            'investor_takeaway': self.investor_takeaway,
            'terms_explained': [{'term': t.term, 'definition': t.definition, 'source': t.source}
                               for t in self.terms_explained],
            'difficulty': self.difficulty.value,
            'difficulty_reason': self.difficulty_reason,
            'educational_summary': self.educational_summary,
            'sources': [{'source_type': s.source_type, 'details': s.details} for s in self.sources],
            'confidence_score': self.confidence_score,
            'sentiment_analysis': {
                'positive_pct': self.sentiment_analysis.positive_pct,
                'neutral_pct': self.sentiment_analysis.neutral_pct,
                'negative_pct': self.sentiment_analysis.negative_pct,
                'total_chunks': self.sentiment_analysis.total_chunks,
                'overall_tone': self.sentiment_analysis.overall_tone
            } if self.sentiment_analysis else None
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class EducationalBriefFormatter:
    """Formats EducationalBrief for display."""
    
    def format_text(self, brief: EducationalBrief) -> str:
        """Format brief as readable text output."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"FinBrief: Educational Investment Brief — {brief.company_name} ({brief.ticker})")
        lines.append(f"Analysis Date: {brief.analysis_date}")
        lines.append("=" * 80)
        lines.append("")
        
        # 1. Company Summary
        lines.append("1. Company Summary (Beginner-Friendly)")
        lines.append("-" * 40)
        lines.append(self._wrap_text(brief.company_summary))
        if brief.summary_citations:
            lines.append("")
            lines.append("Citations:")
            for citation in brief.summary_citations:
                lines.append(f"  {citation}")
        lines.append("")
        
        # 2. Key Financial Metrics
        if brief.metrics:
            lines.append("2. Key Financial Metrics for Students")
            lines.append("-" * 40)
            lines.append("(from Finnhub API)")
            lines.append("")
            for metric in brief.metrics:
                lines.append(f"  • {metric.name}: {metric.value}")
            if brief.metrics_interpretation:
                lines.append("")
                lines.append("Student interpretation:")
                lines.append(self._wrap_text(brief.metrics_interpretation, indent=2))
            lines.append("")
        
        # 3. Opportunities
        if brief.opportunities:
            lines.append("3. Opportunities (Based on SEC filings)")
            lines.append("-" * 40)
            for opp in brief.opportunities:
                category = f" [{opp.category}]" if opp.category else ""
                lines.append(f"  • {opp.description}{category} {opp.citation}")
            lines.append("")
        
        # 4. Risks
        if brief.risks:
            lines.append("4. Risks Students Should Understand")
            lines.append("-" * 40)
            for risk in brief.risks:
                severity_icon = self._get_severity_icon(risk.severity)
                lines.append(f"  {severity_icon} {risk.description} {risk.citation}")
                if risk.explanation:
                    lines.append(f"      → {risk.explanation}")
            lines.append("")
            lines.append("Risk Levels:")
            for risk in brief.risks:
                short_desc = risk.description[:40] + "..." if len(risk.description) > 40 else risk.description
                lines.append(f"  • {short_desc}: {risk.severity.value}")
            lines.append("")
        
        # 5. What This Means for a New Investor
        if brief.investor_takeaway:
            lines.append("5. What This Means for a New Investor")
            lines.append("-" * 40)
            lines.append(self._wrap_text(brief.investor_takeaway))
            lines.append("")
        
        # 6. Key Terms Explained
        if brief.terms_explained:
            lines.append("6. Key Terms Explained (Using Definitions Corpus)")
            lines.append("-" * 40)
            for term in brief.terms_explained:
                lines.append(f"  • {term.term}: {term.definition}")
            lines.append("")
        
        # 7. Beginner Difficulty Score
        lines.append(f"7. Beginner Difficulty Score: {brief.difficulty.value}")
        lines.append("-" * 40)
        if brief.difficulty_reason:
            lines.append(f"Reason: {brief.difficulty_reason}")
        lines.append("")
        
        # 8. Educational Summary
        if brief.educational_summary:
            lines.append("8. Educational Summary (Not Investment Advice)")
            lines.append("-" * 40)
            lines.append(self._wrap_text(brief.educational_summary))
            lines.append("")
        
        # 9. Retrieval Sources
        if brief.sources:
            lines.append("9. Retrieval Sources")
            lines.append("-" * 40)
            for source in brief.sources:
                lines.append(f"  • {source.source_type}: {source.details}")
            lines.append("")
        
        # 10. Confidence Score (optional)
        if brief.confidence_score is not None:
            lines.append(f"10. Model-Estimated Confidence: {brief.confidence_score:.2f}")
            lines.append("")
        
        lines.append("=" * 80)
        lines.append("This is educational content for learning purposes only.")
        lines.append("    It is NOT investment advice. Always do your own research.")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def format_markdown(self, brief: EducationalBrief) -> str:
        """Format brief as Markdown."""
        lines = []
        
        lines.append(f"# FinBrief: Educational Investment Brief — {brief.company_name} ({brief.ticker})")
        lines.append(f"*Analysis Date: {brief.analysis_date}*")
        lines.append("")
        
        # 1. Company Summary
        lines.append("## 1. Company Summary (Beginner-Friendly)")
        lines.append(brief.company_summary)
        if brief.summary_citations:
            lines.append("")
            lines.append("**Citations:**")
            for citation in brief.summary_citations:
                lines.append(f"- {citation}")
        lines.append("")
        
        # 2. Metrics
        if brief.metrics:
            lines.append("## 2. Key Financial Metrics for Students")
            lines.append("*(from Finnhub API)*")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for m in brief.metrics:
                lines.append(f"| {m.name} | {m.value} |")
            if brief.metrics_interpretation:
                lines.append("")
                lines.append(f"**Student interpretation:** {brief.metrics_interpretation}")
            lines.append("")
        
        # 3. Opportunities
        if brief.opportunities:
            lines.append("## 3. Opportunities (Based on SEC filings)")
            for opp in brief.opportunities:
                lines.append(f"- {opp.description} *{opp.citation}*")
            lines.append("")
        
        # 4. Risks
        if brief.risks:
            lines.append("## 4. Risks Students Should Understand")
            for risk in brief.risks:
                icon = self._get_severity_icon(risk.severity)
                lines.append(f"- {icon} **{risk.severity.value}**: {risk.description} *{risk.citation}*")
            lines.append("")
        
        # 5. Investor Takeaway
        if brief.investor_takeaway:
            lines.append("## 5. What This Means for a New Investor")
            lines.append(brief.investor_takeaway)
            lines.append("")
        
        # 6. Terms
        if brief.terms_explained:
            lines.append("## 6. Key Terms Explained")
            for term in brief.terms_explained:
                lines.append(f"- **{term.term}**: {term.definition}")
            lines.append("")
        
        # 7. Difficulty
        lines.append(f"## 7. Beginner Difficulty Score: {brief.difficulty.value}")
        if brief.difficulty_reason:
            lines.append(f"*{brief.difficulty_reason}*")
        lines.append("")
        
        # 8. Educational Summary
        if brief.educational_summary:
            lines.append("## 8. Educational Summary")
            lines.append(f"> **Not Investment Advice**")
            lines.append("")
            lines.append(brief.educational_summary)
            lines.append("")
        
        # 9. Sources
        if brief.sources:
            lines.append("## 9. Retrieval Sources")
            for source in brief.sources:
                lines.append(f"- **{source.source_type}**: {source.details}")
            lines.append("")
        
        # 10. Confidence
        if brief.confidence_score is not None:
            lines.append(f"## 10. Model-Estimated Confidence: {brief.confidence_score:.2f}")
            lines.append("")
        
        lines.append("---")
        lines.append("*This is educational content for learning purposes only. It is NOT investment advice.*")
        
        return "\n".join(lines)
    
    def _wrap_text(self, text: str, width: int = 76, indent: int = 0) -> str:
        """Word-wrap text to specified width."""
        if not text:
            return ""
        
        words = text.split()
        lines = []
        current_line = " " * indent
        
        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                current_line += (" " if current_line.strip() else "") + word
            else:
                if current_line.strip():
                    lines.append(current_line)
                current_line = " " * indent + word
        
        if current_line.strip():
            lines.append(current_line)
        
        return "\n".join(lines)
    
    def _get_severity_icon(self, severity: RiskSeverity) -> str:
        """Get icon for risk severity."""
        icons = {
            RiskSeverity.HIGH: "🔴",
            RiskSeverity.MEDIUM_HIGH: "🟠",
            RiskSeverity.MEDIUM: "🟡",
            RiskSeverity.LOW: "🟢"
        }
        return icons.get(severity, "⚪")


# Example usage / testing
if __name__ == "__main__":
    # Create a sample brief matching the design.md example
    brief = EducationalBrief(
        ticker="AAPL",
        company_name="Apple Inc.",
        analysis_date="2024-12-05",
        company_summary=(
            "Apple designs and sells consumer electronics such as the iPhone, Mac, and iPad. "
            "The company generates most of its revenue from hardware, but Services (iCloud, "
            "App Store, AppleCare) have grown steadily. Apple's financial statements show "
            "consistent profitability, strong cash reserves, and large-scale share repurchases."
        ),
        summary_citations=["(10-K Item 1: Business)", "(10-K Item 7: MD&A)"],
        metrics=[
            FinancialMetric("Market Cap", "$3.2T"),
            FinancialMetric("P/E Ratio", "29.1"),
            FinancialMetric("Revenue Growth (YoY)", "+7.8%"),
            FinancialMetric("EPS (TTM)", "$6.42"),
            FinancialMetric("Debt-to-Equity", "1.78"),
        ],
        metrics_interpretation=(
            "A higher P/E ratio means the stock is relatively expensive compared to earnings. "
            "Apple's P/E is above the S&P 500 average, which suggests investors expect continued growth."
        ),
        opportunities=[
            OpportunityItem(
                "Growth in the Services segment continues to raise margins",
                "(10-K Item 7)",
                "Services Growth"
            ),
            OpportunityItem(
                "Expansion in emerging markets increases long-term customer acquisition",
                "(10-K Item 1A)",
                "Market Expansion"
            ),
            OpportunityItem(
                "Large cash reserves support continued buybacks and R&D",
                "(10-K Financial Statements)",
                "Capital Allocation"
            ),
        ],
        risks=[
            RiskItem(
                "Supply chain disruptions may reduce hardware shipments",
                RiskSeverity.MEDIUM,
                "(10-K Item 1A)"
            ),
            RiskItem(
                "High dependence on the iPhone line creates concentration risk",
                RiskSeverity.MEDIUM_HIGH,
                "(10-K Item 7)"
            ),
            RiskItem(
                "Regulatory pressure and antitrust investigations may limit services revenue",
                RiskSeverity.HIGH,
                "(10-K Item 1A)"
            ),
        ],
        investor_takeaway=(
            "Apple is financially stable, but its valuation is already high, which means "
            "future returns depend on continued growth. Beginners often misunderstand that "
            "strong companies do not always make the best investments at any price. Apple is "
            "strong fundamentally, but the stock may be priced for perfection."
        ),
        terms_explained=[
            TermExplanation(
                "Earnings Per Share (EPS)",
                "Profit divided by the number of shares. Shows how much money the company earns per share you could buy.",
                "Investor.gov"
            ),
            TermExplanation(
                "P/E Ratio",
                "Measures how expensive a stock is relative to earnings. High P/E = high expectations.",
                "Investopedia"
            ),
            TermExplanation(
                "Risk Factor",
                "A potential event that could negatively affect performance. Required in SEC filings.",
                "Investor.gov"
            ),
        ],
        difficulty=DifficultyLevel.MODERATE,
        difficulty_reason="Large company with predictable results, but heavy financial terminology in the filings.",
        educational_summary=(
            "Apple remains highly profitable with durable brand power. Students should pay "
            "attention to the relationship between growth in Services and overall valuation, "
            "and how regulatory pressures could affect margin expansion."
        ),
        sources=[
            Citation("SEC Filing", "10-K (Sections 1, 1A, 7)"),
            Citation("Definitions", "Investor.gov (EPS, Risk Factor), Investopedia (P/E Ratio)"),
            Citation("Finnhub", "Basic Metrics"),
        ],
        confidence_score=0.84
    )
    
    formatter = EducationalBriefFormatter()
    print(formatter.format_text(brief))

