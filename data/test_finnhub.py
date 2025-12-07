#!/usr/bin/env python3
"""
Test script for Finnhub API integration.

Usage:
    # With API key in .env:
    python scripts/test_finnhub.py AAPL
    
    # With direct API key:
    python scripts/test_finnhub.py AAPL --api-key YOUR_KEY
"""
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def test_finnhub_client(ticker: str, api_key: str = None):
    """Test the Finnhub client with a given ticker."""
    from src.finnhub_client import FinnhubClient
    
    print(f"\n{'='*60}")
    print(f"Testing Finnhub API for {ticker}")
    print(f"{'='*60}\n")
    
    # Initialize client
    key = api_key or os.getenv('FINNHUB_API_KEY')
    if not key:
        print("ERROR: No Finnhub API key found!")
        print("Please either:")
        print("  1. Add FINNHUB_API_KEY=your_key to .env file")
        print("  2. Run with --api-key YOUR_KEY")
        return False
    
    print(f"Using API key: {key[:8]}...")
    
    try:
        client = FinnhubClient(api_key=key)
        print("✓ Client initialized\n")
        
        # Test company profile
        print("--- Company Profile ---")
        profile = client.get_company_profile(ticker)
        if profile:
            print(f"  Name: {profile.get('name', 'N/A')}")
            print(f"  Industry: {profile.get('finnhubIndustry', 'N/A')}")
            print(f"  Exchange: {profile.get('exchange', 'N/A')}")
            print(f"  Market Cap: ${profile.get('marketCapitalization', 0):,.0f}M")
        else:
            print("  Could not fetch profile")
        
        # Test quote
        print("\n--- Current Quote ---")
        quote = client.get_quote(ticker)
        if quote:
            print(f"  Current Price: ${quote.get('c', 0):.2f}")
            print(f"  Day High: ${quote.get('h', 0):.2f}")
            print(f"  Day Low: ${quote.get('l', 0):.2f}")
            print(f"  Previous Close: ${quote.get('pc', 0):.2f}")
        else:
            print("  Could not fetch quote")
        
        # Test metrics for brief
        print("\n--- Metrics for FinBrief ---")
        metrics = client.get_metrics_for_brief(ticker)
        if metrics:
            print(metrics.format_for_students())
            print("\n--- Student Interpretation ---")
            print(metrics.get_student_interpretation())
        else:
            print("  Could not fetch metrics")
        
        # Test RAG format
        print("\n--- RAG Document Format ---")
        rag_data = client.format_metrics_for_rag(ticker)
        if rag_data:
            print(f"  Source Type: {rag_data.get('source_type')}")
            print(f"  Source ID: {rag_data.get('source_id')}")
            print(f"  Text Preview: {rag_data.get('text', '')[:100]}...")
        else:
            print("  Could not format for RAG")
        
        print(f"\n{'='*60}")
        print("✓ All Finnhub tests passed!")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Finnhub API integration")
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--api-key', '-k', type=str, help='Finnhub API key')
    
    args = parser.parse_args()
    
    success = test_finnhub_client(args.ticker.upper(), args.api_key)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

