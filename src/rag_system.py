"""
Core RAG system for financial analysis, uses sentence embeddings + FAISS for semantic retrieval from SEC filings.

Core RAG logic (vector embeddings, similarity search, retrieval) designed and implemented by myself
with some AI assistance (logging, fallback handling, and utility functions).
"""
import os
import gc
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dataclasses import dataclass, field


def clear_memory():
    gc.collect()


@dataclass
class DocumentChunk:
    """Text chunk with metadata."""
    text: str
    source_type: str  
    source_id: str
    source_url: str
    filing_date: str = ""
    chunk_index: int = 0
    source_name: str = ""
    is_trusted: bool = False
    term_name: str = "" 
    category: str = ""
    section: str = ""  


class RAGSystem:
    """Core RAG system."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        print(f"loading embedding model: {model_name}")
        # load sentence transformer for text embeddings
        self.embedding_model = SentenceTransformer(model_name, device='cpu')
        self.embedding_dim = embedding_dim
        # FAISS index for vector search
        self.index = None
        self.documents: List[DocumentChunk] = []
        self.index_built = False
        self.max_documents = 500
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for document processing."""
        text_length = len(text)
        if text_length <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        max_chunks = 20
        
        while start < text_length:
            if len(chunks) >= max_chunks:
                break
            
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def _infer_section_from_content(self, text: str) -> str:
        """
        Infer SEC section from content using keyword matching.
        AI-generated: keyword lists and scoring logic.
        """
        text_lower = text.lower()
        
        risk_keywords = ['risk', 'uncertainty', 'vulnerabilities', 'threats', 'challenges', 
                        'adverse', 'volatility', 'exposure', 'dependent', 'could adversely']
        
        business_keywords = ['business', 'products', 'services', 'operations', 'segments',
                            'markets', 'customers', 'revenue', 'principal products']
        
        mda_keywords = ['results of operations', 'financial condition', 'liquidity', 
                       'cash flows', 'capital resources', 'outlook', 'management believes']
        
        risk_score = sum(1 for kw in risk_keywords if kw in text_lower)
        business_score = sum(1 for kw in business_keywords if kw in text_lower)
        mda_score = sum(1 for kw in mda_keywords if kw in text_lower)
        
        scores = {
            'item_1a': risk_score,
            'item_1': business_score,
            'item_7': mda_score
        }
        
        max_section = max(scores, key=scores.get)
        
        if scores[max_section] >= 2:
            return max_section
        
        return 'item_1'  
    
    def add_sec_filings(self, filings: List[Dict]):
        """
        Add SEC filings to document store. AI-generated: fallback logic and section extraction handling.
        """
        for filing_idx, filing in enumerate(filings):
            content = filing.get('content', '')
            if not content:
                continue

            max_content_length = 200000
            if len(content) > max_content_length:
                content = content[:max_content_length]

            sections_extracted = {}

            if 'section' in filing and filing['section']:
                section_name = filing['section']
                section_mapping = {
                    'business': 'item_1',
                    'risk_factors': 'item_1a',
                    'management_discussion': 'item_7',
                    'market_risk': 'item_7a',
                    'financial_statements': 'item_8'
                }
                item_code = section_mapping.get(section_name, section_name)
                sections_extracted[item_code] = content
                print(f"using validated section: {section_name} -> {item_code} ({len(content):,} chars)")

            elif 'sections' in filing and filing['sections']:
                sections_extracted = filing['sections']
                total_chars = sum(len(s) for s in sections_extracted.values())
                print(f"using pre-extracted sections from edgartools:")
                for sec_name, sec_content in sections_extracted.items():
                    print(f"  {sec_name}: {len(sec_content):,} chars")
                print(f"  total: {len(sections_extracted)} sections, {total_chars:,} chars")

            else:
                print(f"no pre-extracted sections, attempting manual extraction...")
                from .clients.sec_edgar_client import SECEdgarClient
                sec_client = SECEdgarClient()
                section_list = ['item_1', 'item_1a', 'item_7', 'item_7a', 'item_8']

                for section in section_list:
                    section_content = sec_client.extract_section(content, section)
                    if section_content and len(section_content) > 100:
                        sections_extracted[section] = section_content

                if sections_extracted:
                    print(f"manual extraction succeeded: {list(sections_extracted.keys())}")
                    for sec_name, sec_content in sections_extracted.items():
                        print(f"  {sec_name}: {len(sec_content):,} chars")
                else:
                    print(f"warning: manual extraction failed")

            # chunk sections into index
            if sections_extracted:
                print(f"\nchunking sections into rag index")
                section_chunk_counts = {}
                docs_before = len(self.documents)

                for section_name, section_content in sections_extracted.items():
                    chunks = self.chunk_text(section_content, chunk_size=800, overlap=150)
                    section_chunk_counts[section_name] = len(chunks)

                    print(f"  {section_name}: {len(section_content):,} chars -> {len(chunks)} chunks")

                    for chunk_idx, chunk in enumerate(chunks):
                        if len(self.documents) >= self.max_documents:
                            print(f"document limit reached ({self.max_documents}), skipping remaining")
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
                print(f"\nadded to rag index: {docs_added} chunks from {len(sections_extracted)} sections")
                print(f"rag index now has {len(self.documents)} total chunks\n")
            else:
                # fallback option - chunk entire document
                print(f"fallback: chunking entire document (lower quality)")
                chunks = self.chunk_text(content, chunk_size=800, overlap=150)
                print(f"created {len(chunks)} chunks from full document")

                for chunk_idx, chunk in enumerate(chunks):
                    if len(self.documents) >= self.max_documents:
                        print(f"document limit reached")
                        return

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

                print(f"using keyword-based section inference\n")
    
    def add_financial_metrics(self, metrics_data: Dict):
        """
        Add financial metrics (Finnhub) to document store.
        Generated with help of AI assistance.
        """
        if len(self.documents) >= self.max_documents:
            print("document limit reached")
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
        """Build FAISS index from documents."""
        if not self.documents:
            print("no documents to index")
            return
        
        print(f"building index for {len(self.documents)} chunks...")
        
        # generate embeddings for all document chunks
        texts = [doc.text for doc in self.documents]
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size
            batch = texts[i:batch_end]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)
            clear_memory()
        
        embeddings = np.vstack(all_embeddings)
        
        # create FAISS index for vector similarity search
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        # normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.index_built = True
        print(f"index built with {self.index.ntotal} vectors")
        
        del embeddings
        del all_embeddings
        clear_memory()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve top-k most relevant chunks using cosine similarity."""
        if not self.index_built:
            raise ValueError("index not built, call build_index() first")
        if self.index is None:
            raise ValueError("index not built, call build_index() first")
        
        # generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        # normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # search for most similar document vectors
        query_vec = query_embedding.astype('float32')
        distances, indices = self.index.search(query_vec, top_k)
        
        # convert distances to similarity scores
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= len(self.documents):
                continue
            # convert L2 distance to similarity score
            similarity = 1 - (distance / 2)
            doc = self.documents[idx]
            results.append((doc, float(similarity)))
        
        return results
    
    def get_context_with_citations(self, query: str, top_k: int = None,
                                     source_types: Optional[List[str]] = None,
                                     purpose: str = None,
                                     min_coverage: float = 0.3,
                                     ensure_all_sections: bool = False) -> Tuple[str, List[Dict]]:
        """
        Retrieve context with citations. Adaptive top_k calculation and balanced section sampling.
        AI-generated: adaptive retrieval logic, section balancing, and logging.
        """
        if top_k is None:
            if source_types:
                available_chunks = len([d for d in self.documents if d.source_type in source_types])
            else:
                available_chunks = len(self.documents)

            top_k = max(5, int(available_chunks * min_coverage))
            top_k = min(top_k, 50)  # cap at 50

            print(f"adaptive retrieval: {available_chunks} chunks available, retrieving top_k={top_k}")

        retrieved = self.retrieve(query, top_k=top_k * 2 if source_types else top_k)

        min_similarity = 0.3
        retrieved = [(doc, score) for doc, score in retrieved if score >= min_similarity]

        if source_types:
            retrieved = [(doc, score) for doc, score in retrieved if doc.source_type in source_types]
            retrieved = retrieved[:top_k]

        if ensure_all_sections and len(retrieved) > 10:
            section_buckets = {}
            for doc, score in retrieved:
                section = doc.section if hasattr(doc, 'section') else ''
                if section and section not in section_buckets:
                    section_buckets[section] = []
                if section:
                    section_buckets[section].append((doc, score))

            min_per_section = max(3, int(top_k * 0.2))

            balanced = []
            for section, chunks in section_buckets.items():
                section_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
                balanced.extend(section_chunks[:min_per_section])

            remaining_slots = top_k - len(balanced)
            if remaining_slots > 0:
                remaining = [c for c in retrieved if c not in balanced]
                balanced.extend(remaining[:remaining_slots])

            retrieved = balanced[:top_k]
            print(f"balanced sampling: {min_per_section}+ chunks per section")

        purpose_label = f" - {purpose}" if purpose else ""
        print(f"\nrag retrieval{purpose_label}")
        print(f"query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"retrieved: {len(retrieved)} chunks")

        if retrieved:
            section_breakdown = {}
            total_chars = 0
            for doc, score in retrieved:
                section = doc.section if hasattr(doc, 'section') else 'unknown'
                if section not in section_breakdown:
                    section_breakdown[section] = []
                section_breakdown[section].append(score)
                total_chars += len(doc.text)

            section_names = {
                'item_1': 'business',
                'item_1a': 'risk factors',
                'item_7': 'md&a',
                'item_7a': 'market risk',
                'item_8': 'financials'
            }

            print(f"chunks by section:")
            for section, scores in sorted(section_breakdown.items()):
                avg_score = sum(scores) / len(scores)
                friendly_name = section_names.get(section, section)
                print(f"  {section} ({friendly_name}): {len(scores)} chunks, avg sim {avg_score:.3f}")

            print(f"total context: {total_chars:,} chars from {len(retrieved)} chunks")

            # coverage analysis
            print(f"coverage:")
            for section in section_breakdown.keys():
                section_docs = [d for d in self.documents if hasattr(d, 'section') and d.section == section]
                if section_docs:
                    retrieved_count = len(section_breakdown[section])
                    coverage_pct = (retrieved_count / len(section_docs)) * 100
                    friendly_name = section_names.get(section, section)

                    if coverage_pct >= 50:
                        status = "good"
                    elif coverage_pct >= 20:
                        status = "moderate"
                    else:
                        status = "low"

                    print(f"  {section} ({friendly_name}): {coverage_pct:.1f}% ({status})")
        else:
            print(f"warning: no chunks retrieved")
        print()

        context_parts = []
        citations = []
        
        for i, (doc, similarity) in enumerate(retrieved):
            citation_id = f"[{i+1}]"
            
            if doc.source_type == 'sec_filing':
                truncated_text = doc.text[:800] if len(doc.text) > 800 else doc.text
            else:
                truncated_text = doc.text[:500] if len(doc.text) > 500 else doc.text
            
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
                'full_text': doc.text,
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
        Find relevant definitions for terms in text.
        AI-generated: definition matching logic.
        """
        definition_docs = [d for d in self.documents if d.source_type == 'definition']
        if not definition_docs:
            return []
        
        text_lower = text.lower()
        scored = []
        
        for doc in definition_docs:
            if doc.term_name:
                term_lower = doc.term_name.lower()
                score = 0
                for word in term_lower.split():
                    if len(word) > 3 and word in text_lower:
                        score += text_lower.count(word)
                
                if score > 0:
                    scored.append((doc, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [{
            'term': doc.term_name,
            'definition': doc.text,
            'source': doc.source_name,
            'url': doc.source_url
        } for doc, _ in scored[:max_terms]]
    
    def clear(self):
        """Clear documents and free memory."""
        self.documents = []
        self.index = None
        self.index_built = False
        clear_memory()
