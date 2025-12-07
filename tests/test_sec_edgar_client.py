"""Tests for the SEC EDGAR client module."""
import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sec_edgar_client import SECEdgarClient, SEC_SECTION_PATTERNS


class TestSECEdgarClient:
    """Test suite for SECEdgarClient."""
    
    @pytest.fixture
    def client(self):
        """Create a client instance for testing."""
        return SECEdgarClient(
            company_name="FinBriefTest",
            name="TestUser",
            email="test@example.com"
        )
    
    def test_client_initialization(self, client):
        """Test client initializes correctly."""
        assert client.user_agent is not None
        assert "FinBriefTest" in client.user_agent
        assert "test@example.com" in client.user_agent
    
    def test_user_agent_format(self, client):
        """Test User-Agent string format."""
        # SEC requires: [Company Name] [Name] [Email]
        parts = client.user_agent.split()
        assert len(parts) >= 3
    
    def test_section_patterns_exist(self):
        """Test that section patterns are defined."""
        assert 'item_1' in SEC_SECTION_PATTERNS
        assert 'item_1a' in SEC_SECTION_PATTERNS
        assert 'item_7' in SEC_SECTION_PATTERNS
    
    def test_extract_section_item_1a(self, client):
        """Test section extraction for Item 1A (Risk Factors)."""
        sample_content = """
        ITEM 1A. RISK FACTORS
        
        The following risk factors could materially affect our business:
        
        Competition is intense in our industry.
        
        ITEM 2. PROPERTIES
        
        Our headquarters are located in...
        """
        
        section = client.extract_section(sample_content, 'item_1a')
        assert section is not None
        assert "risk factors" in section.lower() or "competition" in section.lower()
    
    def test_extract_section_not_found(self, client):
        """Test section extraction when section doesn't exist."""
        sample_content = "This is just some random text without SEC sections."
        section = client.extract_section(sample_content, 'item_1a')
        assert section is None
    
    def test_extract_section_invalid_type(self, client):
        """Test section extraction with invalid section type."""
        section = client.extract_section("Some content", 'invalid_section')
        assert section is None


class TestSECEdgarClientIntegration:
    """Integration tests that require network access.
    
    These tests are skipped by default. Run with:
    pytest tests/test_sec_edgar_client.py -v --run-integration
    """
    
    @pytest.fixture
    def client(self):
        """Create a client instance for testing."""
        return SECEdgarClient(
            company_name="FinBriefTest",
            name="TestUser",
            email="test@example.com"
        )
    
    @pytest.mark.skip(reason="Requires network access")
    def test_get_company_cik_apple(self, client):
        """Test CIK lookup for Apple."""
        cik = client.get_company_cik("AAPL")
        assert cik is not None
        assert cik == "0000320193"  # Apple's CIK
    
    @pytest.mark.skip(reason="Requires network access")
    def test_get_quarterly_filings_apple(self, client):
        """Test fetching Apple's 10-Q filings."""
        filings = client.get_quarterly_filings("AAPL", limit=1)
        assert len(filings) > 0
        assert filings[0]['form'] == '10-Q'
    
    @pytest.mark.skip(reason="Requires network access")
    def test_get_annual_filings_apple(self, client):
        """Test fetching Apple's 10-K filings."""
        filings = client.get_annual_filings("AAPL", limit=1)
        assert len(filings) > 0
        assert filings[0]['form'] == '10-K'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

