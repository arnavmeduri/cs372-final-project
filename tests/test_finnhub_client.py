"""Tests for the Finnhub API client module."""
import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.finnhub_client import FinnhubClient, FinancialMetrics


class TestFinancialMetrics:
    """Test suite for FinancialMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating metrics object."""
        metrics = FinancialMetrics(
            ticker="AAPL",
            company_name="Apple Inc.",
            market_cap=3200000000000,  # $3.2T
            pe_ratio=29.1,
            eps_ttm=6.42
        )
        
        assert metrics.ticker == "AAPL"
        assert metrics.company_name == "Apple Inc."
        assert metrics.market_cap == 3200000000000
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = FinancialMetrics(
            ticker="AAPL",
            company_name="Apple Inc.",
            pe_ratio=29.1
        )
        
        d = metrics.to_dict()
        assert d['ticker'] == "AAPL"
        assert d['pe_ratio'] == 29.1
        assert d['market_cap'] is None  # Not provided
    
    def test_format_for_students_trillion(self):
        """Test formatting market cap in trillions."""
        # Finnhub returns market cap in millions, so $3.2T = 3,200,000 (millions)
        metrics = FinancialMetrics(
            ticker="AAPL",
            company_name="Apple Inc.",
            market_cap=3200000,  # $3.2T in millions
            pe_ratio=29.1
        )
        
        output = metrics.format_for_students()
        assert "AAPL" in output
        assert "$3.20T" in output
        assert "29.1" in output
    
    def test_format_for_students_billion(self):
        """Test formatting market cap in billions."""
        # Finnhub returns market cap in millions, so $150B = 150,000 (millions)
        metrics = FinancialMetrics(
            ticker="IBM",
            company_name="IBM Corporation",
            market_cap=150000  # $150B in millions
        )
        
        output = metrics.format_for_students()
        assert "$150.00B" in output
    
    def test_format_for_students_negative_growth(self):
        """Test formatting negative revenue growth."""
        metrics = FinancialMetrics(
            ticker="TEST",
            company_name="Test Corp",
            revenue_growth_yoy=-5.5
        )
        
        output = metrics.format_for_students()
        assert "-5.5%" in output
    
    def test_student_interpretation_high_pe(self):
        """Test interpretation for high P/E ratio."""
        metrics = FinancialMetrics(
            ticker="GROWTH",
            company_name="Growth Corp",
            pe_ratio=45.0
        )
        
        interpretation = metrics.get_student_interpretation()
        assert "45.0" in interpretation
        assert "high" in interpretation.lower() or "expensive" in interpretation.lower()
    
    def test_student_interpretation_low_pe(self):
        """Test interpretation for low P/E ratio."""
        metrics = FinancialMetrics(
            ticker="VALUE",
            company_name="Value Corp",
            pe_ratio=8.0
        )
        
        interpretation = metrics.get_student_interpretation()
        assert "8.0" in interpretation
        assert "low" in interpretation.lower()
    
    def test_student_interpretation_high_debt(self):
        """Test interpretation for high debt-to-equity."""
        metrics = FinancialMetrics(
            ticker="DEBT",
            company_name="Debt Corp",
            debt_to_equity=3.5
        )
        
        interpretation = metrics.get_student_interpretation()
        assert "3.5" in interpretation or "debt" in interpretation.lower()
        assert "high" in interpretation.lower() or "risk" in interpretation.lower()
    
    def test_student_interpretation_high_beta(self):
        """Test interpretation for high beta."""
        metrics = FinancialMetrics(
            ticker="VOL",
            company_name="Volatile Corp",
            beta=1.8
        )
        
        interpretation = metrics.get_student_interpretation()
        assert "1.8" in interpretation or "volatile" in interpretation.lower()
    
    def test_student_interpretation_empty(self):
        """Test interpretation with no significant metrics."""
        metrics = FinancialMetrics(
            ticker="EMPTY",
            company_name="Empty Corp"
        )
        
        interpretation = metrics.get_student_interpretation()
        assert "No significant metrics" in interpretation


class TestFinnhubClient:
    """Test suite for FinnhubClient."""
    
    def test_client_requires_api_key(self):
        """Test that client requires API key."""
        # Temporarily unset the env var if present
        original_key = os.environ.pop('FINNHUB_API_KEY', None)
        
        try:
            with pytest.raises(ValueError, match="API key is required"):
                FinnhubClient()
        finally:
            # Restore original key if it existed
            if original_key:
                os.environ['FINNHUB_API_KEY'] = original_key
    
    def test_client_initialization_with_key(self):
        """Test client initializes with API key."""
        client = FinnhubClient(api_key="test_api_key")
        assert client.api_key == "test_api_key"
    
    def test_rate_limiting(self):
        """Test that rate limiting is configured."""
        client = FinnhubClient(api_key="test_key")
        assert client.min_request_interval > 0


class TestFinnhubClientIntegration:
    """Integration tests requiring network access.
    
    These tests are skipped by default. Run with:
    pytest tests/test_finnhub_client.py -v --run-integration
    
    Requires FINNHUB_API_KEY environment variable.
    """
    
    @pytest.fixture
    def client(self):
        """Create client if API key is available."""
        api_key = os.environ.get('FINNHUB_API_KEY')
        if not api_key:
            pytest.skip("FINNHUB_API_KEY not set")
        return FinnhubClient(api_key=api_key)
    
    @pytest.mark.skip(reason="Requires network access and API key")
    def test_get_metrics_for_apple(self, client):
        """Test fetching Apple metrics."""
        metrics = client.get_metrics_for_brief("AAPL")
        
        assert metrics is not None
        assert metrics.ticker == "AAPL"
        assert metrics.company_name is not None
        assert metrics.market_cap is not None or metrics.pe_ratio is not None
    
    @pytest.mark.skip(reason="Requires network access and API key")
    def test_format_metrics_for_rag(self, client):
        """Test formatting metrics for RAG ingestion."""
        rag_data = client.format_metrics_for_rag("AAPL")
        
        assert rag_data is not None
        assert 'text' in rag_data
        assert 'source_type' in rag_data
        assert rag_data['source_type'] == 'financial_metrics'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

