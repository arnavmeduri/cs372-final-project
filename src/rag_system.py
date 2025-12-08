"""
RAG (Retrieval-Augmented Generation) system for FinBrief educational financial analysis.
Uses vector embeddings to retrieve relevant sections from SEC filings and financial metrics.
Optimized for Apple Silicon Macs with limited memory.

Data Sources:
- SEC EDGAR filings (10-K, 10-Q, 8-K)
- Finnhub financial metrics
"""
import os
import gc
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dataclasses import dataclass, field


def clear_memory():
    """Run garbage collection."""
    gc.collect()


@dataclass
class DocumentChunk:
    """Represents a chunk of text with metadata."""
    text: str
    source_type: str  # 'sec_filing', 'news', 'financial_metrics'
    source_id: str
    source_url: str
    filing_date: str = ""
    chunk_index: int = 0
    source_name: str = ""  # Name of the source
    is_trusted: bool = False  # Whether from a trusted source
    # Additional metadata for definitions
    term_name: str = ""  # For definitions: the term being defined
    category: str = ""  # Category (e.g., 'valuation', 'risk', 'profitability')
    # Additional metadata for SEC filings
    section: str = ""  # SEC section (e.g., 'item_1', 'item_1a', 'item_7')


class RAGSystem:
    """RAG system for retrieving relevant documents with citations."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        """
        Initialize RAG system with memory-efficient settings.
        
        Args:
            model_name: Name of the sentence transformer model
            embedding_dim: Dimension of embeddings
        """
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name, device='cpu')
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents: List[DocumentChunk] = []
        self.index_built = False
        
        # Limit documents for memory efficiency
        self.max_documents = 500
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        max_chunks = 20  # Limit chunks per document for memory
        
        while start < len(text) and len(chunks) < max_chunks:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def _infer_section_from_content(self, text: str) -> str:
        """
        Infer the SEC section from content using keywords.
        
        Args:
            text: Chunk text to analyze
            
        Returns:
            Inferred section name (e.g., 'item_1a', 'item_1', 'item_7')
        """
        text_lower = text.lower()
        
        # Risk-related keywords
        risk_keywords = ['risk', 'uncertainty', 'vulnerabilities', 'threats', 'challenges', 
                        'adverse', 'volatility', 'exposure', 'dependent', 'could adversely']
        
        # Business description keywords
        business_keywords = ['business', 'products', 'services', 'operations', 'segments',
                            'markets', 'customers', 'revenue', 'principal products']
        
        # Financial/MD&A keywords
        mda_keywords = ['results of operations', 'financial condition', 'liquidity', 
                       'cash flows', 'capital resources', 'outlook', 'management believes']
        
        # Count keyword matches
        risk_score = sum(1 for kw in risk_keywords if kw in text_lower)
        business_score = sum(1 for kw in business_keywords if kw in text_lower)
        mda_score = sum(1 for kw in mda_keywords if kw in text_lower)
        
        # Return section with highest score
        scores = {
            'item_1a': risk_score,
            'item_1': business_score,
            'item_7': mda_score
        }
        
        max_section = max(scores, key=scores.get)
        
        # Only return if we have some confidence (at least 2 keyword matches)
        if scores[max_section] >= 2:
            return max_section
        
        # Default to item_1 if unclear
        return 'item_1'
    
    def add_sec_filings(self, filings: List[Dict]):
        """
        Add SEC filings to the document store with section extraction.

        Args:
            filings: List of filing dictionaries with 'content', 'filing_date', 'url', etc.
                    EdgarToolsClient filings include pre-extracted 'sections' dict.
        """
        for filing_idx, filing in enumerate(filings):
            content = filing.get('content', '')

            # NOTE: We now use edgartools which returns clean, pre-extracted sections
            # No need to skip XBRL prefix - edgartools has already removed it
            # XBRL_PREFIX_LENGTH = 25000  # REMOVED - not needed with edgartools

            if not content:
                continue

            # Limit content length for memory efficiency
            # Increased from 50K to 200K to capture comprehensive SEC filing content
            # Modern LLMs support 128K+ tokens, so 200K chars (~50K tokens) is safe
            max_content_length = 200000
            if len(content) > max_content_length:
                content = content[:max_content_length]

            # Try to use pre-extracted sections first
            sections_extracted = {}

            # Case 1: Single section filing (from generate_rich_analysis validated sections)
            # These have 'section' key (singular) indicating it's already a single section
            if 'section' in filing and filing['section']:
                section_name = filing['section']
                # Map friendly names to item codes
                section_mapping = {
                    'business': 'item_1',
                    'risk_factors': 'item_1a',
                    'management_discussion': 'item_7',
                    'market_risk': 'item_7a',
                    'financial_statements': 'item_8'
                }
                item_code = section_mapping.get(section_name, section_name)
                sections_extracted[item_code] = content
                print(f"✅ [PATH 1] Using validated single section: {section_name} → {item_code} ({len(content):,} chars)")

            # Case 2: Pre-extracted sections (from EdgarToolsClient)
            elif 'sections' in filing and filing['sections']:
                sections_extracted = filing['sections']
                total_chars = sum(len(s) for s in sections_extracted.values())
                print(f"✅ [PATH 2] Using pre-extracted sections from EdgarToolsClient:")
                for sec_name, sec_content in sections_extracted.items():
                    print(f"    - {sec_name}: {len(sec_content):,} chars")
                print(f"    Total: {len(sections_extracted)} sections, {total_chars:,} chars")

            # Case 3: Legacy Fallback - Manual extraction (old SECEdgarClient or missing sections)
            else:
                print(f"⚠️  [PATH 3] LEGACY FALLBACK - No pre-extracted sections found!")
                print(f"    This happens when:")
                print(f"    - Using old SECEdgarClient (pre-edgartools)")
                print(f"    - EdgarToolsClient failed to extract sections")
                print(f"    - Filing format is unusual/unsupported")
                print(f"    Attempting manual regex-based extraction (unreliable)...")

                from .sec_edgar_client import SECEdgarClient
                sec_client = SECEdgarClient()

                section_list = ['item_1', 'item_1a', 'item_7', 'item_7a', 'item_8']

                for section in section_list:
                    section_content = sec_client.extract_section(content, section)
                    if section_content and len(section_content) > 100:  # Valid section
                        sections_extracted[section] = section_content

                if sections_extracted:
                    print(f"✅ Manual extraction succeeded: {list(sections_extracted.keys())}")
                    for sec_name, sec_content in sections_extracted.items():
                        print(f"    - {sec_name}: {len(sec_content):,} chars")
                else:
                    print(f"❌ WARNING: Manual extraction FAILED - no sections found!")
                    print(f"    Expected sections: {section_list}")
                    print(f"    This will result in poor quality analysis!")

            # If we successfully extracted sections, use them
            if sections_extracted:
                print(f"\n{'='*60}")
                print(f"CHUNKING SECTIONS INTO RAG INDEX")
                print(f"{'='*60}")

                section_chunk_counts = {}
                docs_before = len(self.documents)

                for section_name, section_content in sections_extracted.items():
                    chunks = self.chunk_text(section_content, chunk_size=800, overlap=150)
                    section_chunk_counts[section_name] = len(chunks)

                    print(f"  {section_name}:")
                    print(f"    - Content: {len(section_content):,} chars")
                    print(f"    - Chunks: {len(chunks)} × 800 chars (150 char overlap)")

                    for chunk_idx, chunk in enumerate(chunks):
                        if len(self.documents) >= self.max_documents:
                            print(f"⚠️  Document limit ({self.max_documents}) reached. Skipping remaining chunks.")
                            return

                        doc = DocumentChunk(
                            text=chunk,
                            source_type='sec_filing',
                            source_id=f"filing_{filing_idx}",
                            source_url=filing.get('url', ''),
                            filing_date=filing.get('filing_date', ''),
                            chunk_index=chunk_idx,
                            section=section_name
                        )
                        self.documents.append(doc)

                docs_added = len(self.documents) - docs_before
                print(f"\n✅ ADDED TO RAG INDEX:")
                print(f"  - Total sections: {len(sections_extracted)}")
                print(f"  - Total chunks: {docs_added}")
                print(f"  - Breakdown: {dict(section_chunk_counts)}")
                print(f"  - RAG index now has: {len(self.documents)} total chunks")
                print(f"{'='*60}\n")
            else:
                # Extraction failed - chunk entire document and infer sections
                print(f"❌ FALLBACK TO CHUNKING - All section extraction methods failed!")
                print(f"    Will chunk entire document and infer sections from keywords.")
                print(f"    ⚠️  WARNING: This produces LOWER QUALITY results than proper section extraction!")
                print(f"    Document length: {len(content):,} chars")

                chunks = self.chunk_text(content, chunk_size=800, overlap=150)
                print(f"    Created {len(chunks)} chunks from full document")

                for chunk_idx, chunk in enumerate(chunks):
                    if len(self.documents) >= self.max_documents:
                        print(f"⚠️  Document limit ({self.max_documents}) reached. Skipping remaining chunks.")
                        return

                    # Infer section from content
                    inferred_section = self._infer_section_from_content(chunk)

                    doc = DocumentChunk(
                        text=chunk,
                        source_type='sec_filing',
                        source_id=f"filing_{filing_idx}",
                        source_url=filing.get('url', ''),
                        filing_date=filing.get('filing_date', ''),
                        chunk_index=chunk_idx,
                        section=inferred_section
                    )
                    self.documents.append(doc)

                print(f"    ⚠️  Using keyword-based section inference (unreliable)")
    
    def add_news_articles(self, articles: List[Dict]):
        """
        Add news articles to the document store.
        
        Args:
            articles: List of article dictionaries with 'title', 'content', 'url', etc.
        """
        for article_idx, article in enumerate(articles):
            # Combine title and content
            title = article.get('title', '')
            content = article.get('content', '')
            
            # Create clean combined text
            combined = f"{title}\n\n{content}" if content else title
            if not combined.strip():
                continue
            
            chunks = self.chunk_text(combined, chunk_size=800, overlap=150)
            
            for chunk_idx, chunk in enumerate(chunks):
                if len(self.documents) >= self.max_documents:
                    print(f"Warning: Document limit ({self.max_documents}) reached. Skipping remaining articles.")
                    return
                
                doc = DocumentChunk(
                    text=chunk,
                    source_type='news',
                    source_id=f"article_{article_idx}",
                    source_url=article.get('url', ''),
                    filing_date=article.get('published_at', ''),
                    chunk_index=chunk_idx,
                    source_name=article.get('source', 'Unknown'),
                    is_trusted=article.get('is_trusted', False)
                )
                self.documents.append(doc)
    
    def add_financial_metrics(self, metrics_data: Dict):
        """
        Add Finnhub financial metrics to the document store.
        
        Args:
            metrics_data: Dictionary from FinnhubClient.format_metrics_for_rag()
        """
        if len(self.documents) >= self.max_documents:
            print("Warning: Document limit reached. Cannot add metrics.")
            return
        
        doc = DocumentChunk(
            text=metrics_data.get('text', ''),
            source_type='financial_metrics',
            source_id=metrics_data.get('source_id', ''),
            source_url=metrics_data.get('source_url', ''),
            source_name='Finnhub',
            is_trusted=True
        )
        self.documents.append(doc)
    
    def build_index(self):
        """Build FAISS index from all documents with batch processing."""
        if not self.documents:
            print("No documents to index")
            return
        
        print(f"Building index for {len(self.documents)} document chunks...")
        
        # Process in batches for memory efficiency
        texts = [doc.text for doc in self.documents]
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)
            clear_memory()
        
        embeddings = np.vstack(all_embeddings)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.index_built = True
        print(f"Index built with {self.index.ntotal} vectors")
        
        # Clear intermediate data
        del embeddings, all_embeddings
        clear_memory()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve most relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of (DocumentChunk, similarity_score) tuples
        """
        if not self.index_built or self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                # Convert distance to similarity (1 - normalized distance)
                similarity = 1 - (distance / 2)  # L2 distance normalized
                results.append((self.documents[idx], float(similarity)))
        
        return results
    
    def get_context_with_citations(self, query: str, top_k: int = None,
                                     source_types: Optional[List[str]] = None,
                                     purpose: str = None,
                                     min_coverage: float = 0.3,
                                     ensure_all_sections: bool = False) -> Tuple[str, List[Dict]]:
        """
        Get context text and citations for a query with adaptive retrieval.

        Args:
            query: Search query
            top_k: Number of documents to retrieve. If None, calculates adaptively based on content size.
            source_types: Optional filter for source types (e.g., ['sec_filing', 'definition'])
            purpose: Optional description of what this retrieval is for (e.g., "RISKS", "OPPORTUNITIES")
            min_coverage: Minimum coverage percentage (0.0-1.0). Default 0.3 = 30% coverage.
                         Used to calculate top_k if not specified.

        Returns:
            Tuple of (context_text, citations_list)
        """
        # Calculate adaptive top_k if not specified
        if top_k is None:
            # Count available chunks for the requested source types
            if source_types:
                available_chunks = len([d for d in self.documents if d.source_type in source_types])
            else:
                available_chunks = len(self.documents)

            # Calculate top_k to achieve min_coverage
            top_k = max(5, int(available_chunks * min_coverage))  # At least 5, or coverage%
            top_k = min(top_k, 50)  # Cap at 50 to avoid overwhelming LLM context

            print(f"📊 ADAPTIVE RETRIEVAL: {available_chunks} chunks available")
            print(f"   Target coverage: {min_coverage:.0%} → Retrieving top_k={top_k} chunks")

        retrieved = self.retrieve(query, top_k=top_k * 2 if source_types else top_k)

        # Filter by minimum similarity threshold
        MIN_SIMILARITY = 0.3
        retrieved = [(doc, score) for doc, score in retrieved if score >= MIN_SIMILARITY]

        # Filter by source type if specified
        if source_types:
            retrieved = [(doc, score) for doc, score in retrieved if doc.source_type in source_types]
            retrieved = retrieved[:top_k]

        # SECTION-AWARE RETRIEVAL: For comprehensive analysis, ensure ALL sections represented
        if ensure_all_sections and len(retrieved) > 10:
            # Group by section
            section_buckets = {}
            for doc, score in retrieved:
                section = doc.section if hasattr(doc, 'section') else ''
                if section and section not in section_buckets:
                    section_buckets[section] = []
                if section:
                    section_buckets[section].append((doc, score))

            # Calculate minimum chunks per section (at least 20% of total)
            min_per_section = max(3, int(top_k * 0.2))

            # Ensure each section has minimum representation
            balanced = []
            for section, chunks in section_buckets.items():
                # Take best chunks from this section
                section_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
                balanced.extend(section_chunks[:min_per_section])

            # Fill remaining with highest similarity across all sections
            remaining_slots = top_k - len(balanced)
            if remaining_slots > 0:
                remaining = [c for c in retrieved if c not in balanced]
                balanced.extend(remaining[:remaining_slots])

            retrieved = balanced[:top_k]
            print(f"   ⚖️  BALANCED: Ensured {min_per_section}+ chunks per section (3 sections)")

        # Log what was retrieved
        purpose_label = f" - {purpose}" if purpose else ""
        print(f"\n{'='*60}")
        print(f"RAG RETRIEVAL{purpose_label}")
        print(f"{'='*60}")
        print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"Requested: top_k={top_k}, source_types={source_types}")
        print(f"Retrieved: {len(retrieved)} chunks (after filtering)")

        if retrieved:
            section_breakdown = {}
            total_chars = 0
            for doc, score in retrieved:
                section = doc.section if hasattr(doc, 'section') else 'unknown'
                if section not in section_breakdown:
                    section_breakdown[section] = []
                section_breakdown[section].append(score)
                total_chars += len(doc.text)

            # Map section codes to friendly names
            section_names = {
                'item_1': 'Business Description',
                'item_1a': 'Risk Factors',
                'item_7': 'MD&A',
                'item_7a': 'Market Risk',
                'item_8': 'Financial Statements'
            }

            print(f"\nRetrieved chunks by section:")
            for section, scores in sorted(section_breakdown.items()):
                avg_score = sum(scores) / len(scores)
                friendly_name = section_names.get(section, section)
                chunk_chars = len(scores) * 800  # Approximate
                print(f"  - {section} ({friendly_name}):")
                print(f"      {len(scores)} chunks (avg similarity: {avg_score:.3f})")
                print(f"      ~{chunk_chars:,} chars retrieved")

            print(f"\nTotal context: {total_chars:,} chars from {len(retrieved)} chunks")

            # Calculate coverage for each section
            print(f"\nCOVERAGE ANALYSIS:")
            for section in section_breakdown.keys():
                # Find total chars available in this section
                section_docs = [d for d in self.documents if hasattr(d, 'section') and d.section == section]
                if section_docs:
                    # Estimate total section size (chunks × avg chunk size)
                    total_available = len(section_docs) * 800  # Approximate
                    retrieved_count = len(section_breakdown[section])
                    coverage_pct = (retrieved_count / len(section_docs)) * 100
                    friendly_name = section_names.get(section, section)

                    # Determine if coverage is good
                    if coverage_pct >= 50:
                        status = "✅ GOOD"
                    elif coverage_pct >= 20:
                        status = "⚠️  MODERATE"
                    else:
                        status = "❌ LOW"

                    print(f"  {section} ({friendly_name}): {coverage_pct:.1f}% coverage {status}")
                    print(f"    Retrieved {retrieved_count}/{len(section_docs)} chunks available")
        else:
            print(f"⚠️  WARNING: No chunks retrieved! Query may not match indexed content.")
        print(f"{'='*60}\n")

        context_parts = []
        citations = []
        
        for i, (doc, similarity) in enumerate(retrieved):
            citation_id = f"[{i+1}]"
            # Use more text for SEC filings to provide better context (800 chars instead of 500)
            if doc.source_type == 'sec_filing':
                truncated_text = doc.text[:800] if len(doc.text) > 800 else doc.text
            else:
                truncated_text = doc.text[:500] if len(doc.text) > 500 else doc.text
            
            # Add source type indicator for clarity
            source_label = {
                'sec_filing': 'SEC Filing',
                'definition': 'Definition',
                'financial_metrics': 'Metrics',
                'news': 'News'
            }.get(doc.source_type, 'Source')
            
            context_parts.append(f"{citation_id} [{source_label}] {truncated_text}")
            
            citations.append({
                'id': i + 1,
                'text': doc.text[:400] + "..." if len(doc.text) > 400 else doc.text,
                'full_text': doc.text,  # Store full text for hover/expand features
                'source_type': doc.source_type,
                'source_url': doc.source_url,
                'source_name': doc.source_name,
                'is_trusted': doc.is_trusted,
                'date': doc.filing_date,
                'similarity': similarity,
                'term_name': doc.term_name,
                'category': doc.category,
                'section': doc.section
            })
        
        context = "\n\n".join(context_parts)
        return context, citations
    
    def get_definitions_for_text(self, text: str, max_terms: int = 5) -> List[Dict]:
        """
        Find relevant definitions for terms mentioned in text.
        
        Args:
            text: Text to find definitions for
            max_terms: Maximum definitions to return
            
        Returns:
            List of definition dictionaries
        """
        # Get only definition documents
        definition_docs = [d for d in self.documents if d.source_type == 'definition']
        if not definition_docs:
            return []
        
        text_lower = text.lower()
        scored = []
        
        for doc in definition_docs:
            if doc.term_name:
                # Check if term appears in text
                term_lower = doc.term_name.lower()
                # Score based on term presence
                score = 0
                for word in term_lower.split():
                    if len(word) > 3 and word in text_lower:
                        score += text_lower.count(word)
                
                if score > 0:
                    scored.append((doc, score))
        
        # Sort by score and return top matches
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [{
            'term': doc.term_name,
            'definition': doc.text,
            'source': doc.source_name,
            'url': doc.source_url
        } for doc, _ in scored[:max_terms]]
    
    def clear(self):
        """Clear all documents and free memory."""
        self.documents = []
        self.index = None
        self.index_built = False
        clear_memory()
