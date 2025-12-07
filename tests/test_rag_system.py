"""Tests for the RAG system module."""
import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag_system import RAGSystem, DocumentChunk


class TestDocumentChunk:
    """Test suite for DocumentChunk dataclass."""
    
    def test_chunk_creation(self):
        """Test creating a document chunk."""
        chunk = DocumentChunk(
            text="This is a test chunk about financial metrics.",
            source_type="sec_filing",
            source_id="test_1",
            source_url="https://sec.gov/test"
        )
        
        assert chunk.text == "This is a test chunk about financial metrics."
        assert chunk.source_type == "sec_filing"
        assert chunk.is_trusted == False  # Default
    
    def test_chunk_with_all_fields(self):
        """Test chunk with all metadata fields."""
        chunk = DocumentChunk(
            text="Definition text",
            source_type="definition",
            source_id="def_eps",
            source_url="https://investor.gov",
            source_name="Investor.gov",
            is_trusted=True,
            term_name="EPS",
            category="profitability"
        )
        
        assert chunk.term_name == "EPS"
        assert chunk.category == "profitability"
        assert chunk.is_trusted == True


class TestRAGSystem:
    """Test suite for RAGSystem."""
    
    @pytest.fixture
    def rag(self):
        """Create a RAG system instance."""
        return RAGSystem()
    
    def test_initialization(self, rag):
        """Test RAG system initializes correctly."""
        assert rag.embedding_model is not None
        assert rag.embedding_dim == 384
        assert len(rag.documents) == 0
        assert rag.index_built == False
    
    def test_chunk_text_short(self, rag):
        """Test chunking short text."""
        text = "This is a short text."
        chunks = rag.chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_long(self, rag):
        """Test chunking long text."""
        text = "A" * 1500  # 1500 characters
        chunks = rag.chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) > 1
    
    def test_add_sec_filings(self, rag):
        """Test adding SEC filings."""
        filings = [
            {
                'content': 'This is a test SEC filing about company risks.',
                'filing_date': '2024-01-15',
                'url': 'https://sec.gov/test'
            }
        ]
        
        rag.add_sec_filings(filings)
        assert len(rag.documents) > 0
        assert rag.documents[0].source_type == 'sec_filing'
    
    def test_add_news_articles(self, rag):
        """Test adding news articles."""
        articles = [
            {
                'title': 'Test News Article',
                'content': 'This is the content of a news article.',
                'url': 'https://news.com/test',
                'source': 'Reuters',
                'is_trusted': True
            }
        ]
        
        rag.add_news_articles(articles)
        assert len(rag.documents) > 0
        assert rag.documents[0].source_type == 'news'
        assert rag.documents[0].is_trusted == True
    
    def test_add_definitions(self, rag):
        """Test adding definitions."""
        definitions = [
            {
                'text': 'EPS: Earnings Per Share is profit divided by shares.',
                'source_type': 'definition',
                'source_id': 'def_eps',
                'source_url': 'https://investor.gov',
                'source_name': 'Investor.gov',
                'term_name': 'EPS',
                'category': 'profitability'
            }
        ]
        
        rag.add_definitions(definitions)
        assert len(rag.documents) > 0
        assert rag.documents[0].source_type == 'definition'
        assert rag.documents[0].term_name == 'EPS'
    
    def test_add_financial_metrics(self, rag):
        """Test adding financial metrics."""
        metrics = {
            'text': 'Market Cap: $3.2T, P/E Ratio: 29.1',
            'source_id': 'finnhub_AAPL',
            'source_url': 'https://finnhub.io',
            'source_name': 'Finnhub'
        }
        
        rag.add_financial_metrics(metrics)
        assert len(rag.documents) > 0
        assert rag.documents[0].source_type == 'financial_metrics'
    
    def test_build_index(self, rag):
        """Test building FAISS index."""
        # Add some documents
        rag.add_sec_filings([{'content': 'Test content ' * 50, 'url': ''}])
        rag.add_definitions([{
            'text': 'EPS definition',
            'source_id': 'def_1',
            'source_url': '',
            'source_name': 'Test',
            'term_name': 'EPS',
            'category': 'test'
        }])
        
        rag.build_index()
        
        assert rag.index_built == True
        assert rag.index is not None
        assert rag.index.ntotal > 0
    
    def test_retrieve_requires_index(self, rag):
        """Test that retrieve fails without index."""
        with pytest.raises(ValueError, match="Index not built"):
            rag.retrieve("test query")
    
    def test_retrieve_with_index(self, rag):
        """Test retrieval with built index."""
        # Add documents and build index
        rag.add_sec_filings([
            {'content': 'Apple Inc reported strong revenue growth.', 'url': ''},
            {'content': 'Microsoft announced new cloud services.', 'url': ''}
        ])
        rag.build_index()
        
        results = rag.retrieve("Apple revenue", top_k=1)
        assert len(results) > 0
        assert isinstance(results[0], tuple)
        assert isinstance(results[0][0], DocumentChunk)
        assert isinstance(results[0][1], float)
    
    def test_get_context_with_citations(self, rag):
        """Test getting context with citations."""
        rag.add_sec_filings([
            {'content': 'Risk factors include market volatility.', 'url': 'https://sec.gov/1'}
        ])
        rag.build_index()
        
        context, citations = rag.get_context_with_citations("market risk", top_k=1)
        
        assert len(context) > 0
        assert len(citations) > 0
        assert 'id' in citations[0]
        assert 'source_type' in citations[0]
    
    def test_get_context_filter_by_source_type(self, rag):
        """Test filtering context by source type."""
        rag.add_sec_filings([{'content': 'SEC content', 'url': ''}])
        rag.add_definitions([{
            'text': 'Definition content',
            'source_id': 'def_1',
            'source_url': '',
            'source_name': 'Test',
            'term_name': 'Test Term',
            'category': 'test'
        }])
        rag.build_index()
        
        # Filter to only definitions
        context, citations = rag.get_context_with_citations(
            "content", top_k=2, source_types=['definition']
        )
        
        for citation in citations:
            assert citation['source_type'] == 'definition'
    
    def test_clear(self, rag):
        """Test clearing the RAG system."""
        rag.add_sec_filings([{'content': 'Test', 'url': ''}])
        rag.build_index()
        
        rag.clear()
        
        assert len(rag.documents) == 0
        assert rag.index is None
        assert rag.index_built == False
    
    def test_document_limit(self, rag):
        """Test document limit is enforced."""
        # Temporarily set a low limit
        original_limit = rag.max_documents
        rag.max_documents = 5
        
        # Try to add more documents than the limit
        filings = [
            {'content': f'Content {i}' * 100, 'url': ''} 
            for i in range(10)
        ]
        rag.add_sec_filings(filings)
        
        assert len(rag.documents) <= 5
        
        # Restore original limit
        rag.max_documents = original_limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

