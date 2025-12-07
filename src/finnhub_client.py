"""
Finnhub API Client for fetching financial metrics.
Provides simple, student-relevant metrics like P/E ratio, EPS, market cap, etc.

API Documentation: https://finnhub.io/docs/api
Free tier: 60 API calls/minute
"""
import requests
import os
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class FinancialMetrics:
    """Student-friendly financial metrics for FinBrief."""
    ticker: str
    company_name: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    eps_ttm: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    debt_to_equity: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    current_price: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'market_cap': self.market_cap,
            'pe_ratio': self.pe_ratio,
            'eps_ttm': self.eps_ttm,
            'revenue_growth_yoy': self.revenue_growth_yoy,
            'debt_to_equity': self.debt_to_equity,
            'dividend_yield': self.dividend_yield,
            'beta': self.beta,
            'week_52_high': self.week_52_high,
            'week_52_low': self.week_52_low,
            'current_price': self.current_price
        }
    
    def format_for_students(self) -> str:
        """Format metrics in a student-friendly way."""
        lines = [f"**Key Financial Metrics for {self.company_name} ({self.ticker})**\n"]
        
        if self.market_cap:
            # Finnhub returns market cap in millions, so adjust thresholds
            # 1,000,000M = $1T, 1,000M = $1B
            if self.market_cap >= 1e6:  # >= $1T (in millions)
                cap_str = f"${self.market_cap / 1e6:.2f}T"
            elif self.market_cap >= 1e3:  # >= $1B (in millions)
                cap_str = f"${self.market_cap / 1e3:.2f}B"
            else:
                cap_str = f"${self.market_cap:.2f}M"
            lines.append(f"• Market Cap: {cap_str}")
        
        if self.pe_ratio:
            lines.append(f"• P/E Ratio: {self.pe_ratio:.1f}")
        
        if self.eps_ttm:
            lines.append(f"• EPS (TTM): ${self.eps_ttm:.2f}")
        
        if self.revenue_growth_yoy is not None:
            sign = "+" if self.revenue_growth_yoy >= 0 else ""
            lines.append(f"• Revenue Growth (YoY): {sign}{self.revenue_growth_yoy:.1f}%")
        
        if self.debt_to_equity:
            lines.append(f"• Debt-to-Equity: {self.debt_to_equity:.2f}")
        
        if self.dividend_yield:
            lines.append(f"• Dividend Yield: {self.dividend_yield:.2f}%")
        
        if self.beta:
            lines.append(f"• Beta: {self.beta:.2f}")
        
        return "\n".join(lines)
    
    def get_student_interpretation(self) -> str:
        """Generate beginner-friendly interpretation of the metrics."""
        interpretations = []
        
        if self.pe_ratio:
            if self.pe_ratio > 30:
                interpretations.append(
                    f"The P/E ratio of {self.pe_ratio:.1f} is relatively high, suggesting investors "
                    "expect strong future growth. This could mean the stock is expensive."
                )
            elif self.pe_ratio > 15:
                interpretations.append(
                    f"The P/E ratio of {self.pe_ratio:.1f} is moderate, indicating balanced "
                    "expectations for growth."
                )
            else:
                interpretations.append(
                    f"The P/E ratio of {self.pe_ratio:.1f} is relatively low, which could mean "
                    "the stock is undervalued or growth expectations are modest."
                )
        
        if self.debt_to_equity:
            if self.debt_to_equity > 2:
                interpretations.append(
                    f"The debt-to-equity ratio of {self.debt_to_equity:.2f} is high, meaning the company "
                    "uses significant debt financing. This increases financial risk."
                )
            elif self.debt_to_equity < 0.5:
                interpretations.append(
                    f"The debt-to-equity ratio of {self.debt_to_equity:.2f} is low, suggesting "
                    "conservative financing with minimal debt."
                )
        
        if self.beta:
            if self.beta > 1.3:
                interpretations.append(
                    f"With a beta of {self.beta:.2f}, this stock is more volatile than the market. "
                    "It tends to move more dramatically in both directions."
                )
            elif self.beta < 0.8:
                interpretations.append(
                    f"With a beta of {self.beta:.2f}, this stock is less volatile than the market, "
                    "making it potentially more stable but with less upside potential."
                )
        
        if self.revenue_growth_yoy is not None:
            if self.revenue_growth_yoy > 15:
                interpretations.append(
                    f"Revenue growth of {self.revenue_growth_yoy:.1f}% year-over-year is strong, "
                    "indicating the business is expanding rapidly."
                )
            elif self.revenue_growth_yoy < 0:
                interpretations.append(
                    f"Revenue declined {abs(self.revenue_growth_yoy):.1f}% year-over-year, "
                    "which could signal challenges in the business."
                )
        
        return " ".join(interpretations) if interpretations else "No significant metrics to interpret."


class FinnhubClient:
    """
    Client for Finnhub API to fetch financial metrics.
    Free tier: 60 API calls/minute.
    """
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Finnhub client.
        
        Args:
            api_key: Finnhub API key (or from FINNHUB_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Finnhub API key is required. "
                "Get a free key at https://finnhub.io and set FINNHUB_API_KEY in .env"
            )
        
        self.session = requests.Session()
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Rate limiting: max 10 requests/second
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make an API request with rate limiting."""
        self._rate_limit()
        
        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        params['token'] = self.api_key
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 401:
                raise ValueError("Invalid Finnhub API key")
            elif response.status_code == 429:
                print("Rate limit exceeded. Waiting...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Finnhub API error: {e}")
            return None
    
    def get_company_profile(self, ticker: str) -> Optional[Dict]:
        """
        Get company profile information.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Company profile data
        """
        return self._make_request("stock/profile2", {"symbol": ticker})
    
    def get_basic_financials(self, ticker: str) -> Optional[Dict]:
        """
        Get basic financial metrics.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Financial metrics data
        """
        return self._make_request("stock/metric", {"symbol": ticker, "metric": "all"})
    
    def get_quote(self, ticker: str) -> Optional[Dict]:
        """
        Get current stock quote.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Quote data with current price, etc.
        """
        return self._make_request("quote", {"symbol": ticker})
    
    def get_metrics_for_brief(self, ticker: str) -> Optional[FinancialMetrics]:
        """
        Get all metrics needed for a FinBrief educational report.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            FinancialMetrics object with student-relevant data
        """
        # Get company profile
        profile = self.get_company_profile(ticker)
        company_name = profile.get('name', ticker) if profile else ticker
        
        # Get basic financials
        financials = self.get_basic_financials(ticker)
        
        # Get current quote
        quote = self.get_quote(ticker)
        
        if not financials:
            return None
        
        metric = financials.get('metric', {})
        
        return FinancialMetrics(
            ticker=ticker,
            company_name=company_name,
            market_cap=metric.get('marketCapitalization'),
            pe_ratio=metric.get('peBasicExclExtraTTM') or metric.get('peTTM'),
            eps_ttm=metric.get('epsBasicExclExtraItemsTTM') or metric.get('epsTTM'),
            revenue_growth_yoy=metric.get('revenueGrowthTTMYoy'),
            debt_to_equity=metric.get('totalDebt/totalEquityQuarterly'),
            dividend_yield=metric.get('dividendYieldIndicatedAnnual'),
            beta=metric.get('beta'),
            week_52_high=metric.get('52WeekHigh'),
            week_52_low=metric.get('52WeekLow'),
            current_price=quote.get('c') if quote else None
        )
    
    def format_metrics_for_rag(self, ticker: str) -> Optional[Dict]:
        """
        Get metrics formatted as a document chunk for RAG.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary suitable for RAG ingestion
        """
        metrics = self.get_metrics_for_brief(ticker)
        if not metrics:
            return None
        
        text = metrics.format_for_students() + "\n\n" + metrics.get_student_interpretation()
        
        return {
            'text': text,
            'source_type': 'financial_metrics',
            'source_id': f"finnhub_{ticker}",
            'source_url': f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}",
            'source_name': 'Finnhub',
            'ticker': ticker,
            'is_trusted': True,
            'metrics': metrics.to_dict()
        }


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Finnhub Financial Metrics Client")
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--raw', action='store_true', help='Show raw API response')
    
    args = parser.parse_args()
    
    try:
        client = FinnhubClient()
        
        if args.raw:
            print("\n=== Company Profile ===")
            profile = client.get_company_profile(args.ticker)
            print(profile)
            
            print("\n=== Basic Financials ===")
            financials = client.get_basic_financials(args.ticker)
            if financials and 'metric' in financials:
                for key, value in list(financials['metric'].items())[:20]:
                    print(f"  {key}: {value}")
        else:
            metrics = client.get_metrics_for_brief(args.ticker)
            if metrics:
                print(metrics.format_for_students())
                print("\n--- Student Interpretation ---")
                print(metrics.get_student_interpretation())
            else:
                print(f"Could not fetch metrics for {args.ticker}")
                
    except ValueError as e:
        print(f"Error: {e}")

