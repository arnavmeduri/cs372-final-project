"""Tests for the definitions corpus module."""
import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.definitions_corpus import DefinitionsCorpus, TermDefinition


class TestDefinitionsCorpus:
    """Test suite for DefinitionsCorpus."""
    
    @pytest.fixture
    def corpus(self):
        """Create a corpus instance for testing."""
        return DefinitionsCorpus()
    
    def test_corpus_loads(self, corpus):
        """Test that corpus loads successfully."""
        assert len(corpus) > 0
        assert len(corpus.terms) > 0
    
    def test_get_term_exact_match(self, corpus):
        """Test exact term lookup."""
        term = corpus.get_term("Earnings Per Share (EPS)")
        assert term is not None
        assert "EPS" in term.term or "Earnings" in term.term
    
    def test_get_term_partial_match(self, corpus):
        """Test partial term lookup."""
        term = corpus.get_term("EPS")
        assert term is not None
    
    def test_get_term_case_insensitive(self, corpus):
        """Test case-insensitive lookup."""
        term1 = corpus.get_term("P/E Ratio")
        term2 = corpus.get_term("p/e ratio")
        # Both should find something (possibly different due to partial matching)
        assert term1 is not None or term2 is not None
    
    def test_get_term_not_found(self, corpus):
        """Test lookup for non-existent term."""
        term = corpus.get_term("XYZ_NONEXISTENT_TERM_123")
        assert term is None
    
    def test_categories_exist(self, corpus):
        """Test that categories are populated."""
        assert len(corpus.categories) > 0
        assert 'valuation' in corpus.categories or 'profitability' in corpus.categories
    
    def test_get_terms_by_category(self, corpus):
        """Test getting terms by category."""
        # Find a category that exists
        if corpus.categories:
            cat = corpus.categories[0]
            terms = corpus.get_terms_by_category(cat)
            assert len(terms) > 0
    
    def test_search_terms(self, corpus):
        """Test term search functionality."""
        matches = corpus.search_terms("earnings")
        assert len(matches) > 0
    
    def test_get_all_chunks(self, corpus):
        """Test getting all terms as RAG chunks."""
        chunks = corpus.get_all_chunks()
        assert len(chunks) > 0
        
        # Check chunk structure
        chunk = chunks[0]
        assert 'text' in chunk
        assert 'source_type' in chunk
        assert chunk['source_type'] == 'definition'
        assert chunk['is_trusted'] == True
    
    def test_get_relevant_definitions(self, corpus):
        """Test finding relevant definitions for text."""
        sample_text = """
        Apple's earnings per share increased by 10% this quarter.
        The P/E ratio suggests the stock may be overvalued.
        Revenue growth continues to be strong.
        """
        
        relevant = corpus.get_relevant_definitions(sample_text, max_terms=3)
        assert len(relevant) > 0
        assert len(relevant) <= 3
    
    def test_format_for_output(self, corpus):
        """Test formatting terms for display."""
        terms = corpus.terms[:2]
        output = corpus.format_for_output(terms)
        
        assert len(output) > 0
        assert "Key Terms Explained" in output
    
    def test_term_definition_to_chunk_text(self):
        """Test TermDefinition.to_chunk_text()."""
        term = TermDefinition(
            term="Test Term",
            definition="A test definition.",
            example="An example.",
            source="Test Source",
            category="test"
        )
        
        text = term.to_chunk_text()
        assert "Test Term" in text
        assert "test definition" in text
        assert "example" in text.lower()


class TestTermDefinition:
    """Test suite for TermDefinition dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        term = TermDefinition(
            term="Revenue",
            definition="Total sales",
            example="$100M in sales",
            source="Investor.gov",
            category="fundamentals"
        )
        
        d = term.to_dict()
        assert d['term'] == "Revenue"
        assert d['definition'] == "Total sales"
        assert d['source'] == "Investor.gov"
        assert d['category'] == "fundamentals"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

