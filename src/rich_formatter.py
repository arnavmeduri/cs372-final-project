"""
Rich analysis mode formatter.
Displays comprehensive LLM-generated research without rigid structuring.
"""
from .educational_brief import EducationalBrief
from typing import List


class RichAnalysisFormatter:
    """Format rich LLM analysis for display."""
    
    def format_text(self, brief: EducationalBrief) -> str:
        """Format rich analysis as readable text."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"FinBrief Analysis: {brief.company_name} ({brief.ticker})")
        lines.append(f"Analysis Date: {brief.analysis_date}")
        lines.append("=" * 80)
        lines.append("")
        lines.append("RAG-Augmented Research Report")
        lines.append("")
        lines.append("="* 80)
        
        # Main comprehensive analysis
        lines.append("")
        lines.append(brief.company_summary)  # This contains the full rich analysis
        lines.append("")
        
        # Financial metrics sidebar
        if brief.metrics:
            lines.append("")
            lines.append("=" * 80)
            lines.append("KEY METRICS SUMMARY")
            lines.append("=" * 80)
            for metric in brief.metrics:
                lines.append(f"  • {metric.name}: {metric.value}")
            lines.append("")
        
        # Sources section at the end
        if brief.summary_citations:
            lines.append("=" * 80)
            lines.append("SOURCES")
            lines.append("=" * 80)
            for citation in brief.summary_citations:
                lines.append(f"  {citation}")
            lines.append("")
        
        # Footer
        lines.append("=" * 80)
        lines.append("Educational Analysis for Learning Purposes")
        lines.append("   This is NOT investment advice. For educational use only.")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def format_markdown(self, brief: EducationalBrief) -> str:
        """Format rich analysis as markdown."""
        lines = []
        
        # Header
        lines.append(f"# {brief.company_name} ({brief.ticker}) - Research Analysis")
        lines.append(f"*Analysis Date: {brief.analysis_date}*")
        lines.append("")
        lines.append("**RAG-Augmented Research Report**")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Main analysis
        lines.append(brief.company_summary)
        lines.append("")
        
        # Metrics
        if brief.metrics:
            lines.append("---")
            lines.append("")
            lines.append("## Key Metrics Summary")
            lines.append("")
            for metric in brief.metrics:
                lines.append(f"- **{metric.name}**: {metric.value}")
            lines.append("")
        
        # Sources section at the end
        if brief.summary_citations:
            lines.append("---")
            lines.append("")
            lines.append("## Sources")
            lines.append("")
            for citation in brief.summary_citations:
                lines.append(f"- {citation}")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Educational Analysis for Learning Purposes*")
        lines.append("")
        lines.append("*This is NOT investment advice. For educational use only.*")
        
        return "\n".join(lines)


