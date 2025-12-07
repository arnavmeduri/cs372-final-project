#!/usr/bin/env python3
"""
Test FinBrief with any company ticker.

Usage:
    python scripts/test_any_company.py AAPL
    python scripts/test_any_company.py MSFT --with-model
    python scripts/test_any_company.py COST --full
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.finbrief import FinBriefApp
from src.educational_brief import EducationalBriefFormatter


def test_company(ticker: str, use_model: bool = False, full_output: bool = False):
    """Test FinBrief with any company ticker."""
    print(f"\n{'='*80}")
    print(f"Testing FinBrief with: {ticker}")
    print(f"{'='*80}\n")
    
    try:
        app = FinBriefApp()
        
        # Generate brief
        brief = app.generate_brief(
            ticker=ticker,
            filing_type="10-K",
            use_model=use_model
        )
        
        # Format output
        formatter = EducationalBriefFormatter()
        
        if full_output:
            output = formatter.format_markdown(brief)
        else:
            output = formatter.format_text(brief)
        
        print(output)
        
        # Summary stats
        print(f"\n{'='*80}")
        print("SUMMARY STATS")
        print(f"{'='*80}")
        print(f"Company: {brief.company_name} ({brief.ticker})")
        print(f"Risks identified: {len(brief.risks)}")
        print(f"Opportunities identified: {len(brief.opportunities)}")
        print(f"Key terms explained: {len(brief.terms_explained)}")
        print(f"Confidence score: {brief.confidence_score:.2f}" if brief.confidence_score else "N/A")
        print(f"Sources: {len(brief.sources)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FinBrief with any company")
    parser.add_argument('ticker', help='Stock ticker symbol (e.g., AAPL, MSFT, COST)')
    parser.add_argument('--with-model', action='store_true', 
                       help='Use LLM generation (slower but more detailed)')
    parser.add_argument('--full', action='store_true',
                       help='Show full markdown output')
    
    args = parser.parse_args()
    
    success = test_company(
        args.ticker.upper(),
        use_model=args.with_model,
        full_output=args.full
    )
    
    sys.exit(0 if success else 1)

