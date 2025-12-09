"""
SEC EDGAR API Client for fetching company filings and financial reports.
Supports 10-K (annual), 10-Q (quarterly), and 8-K (material events) filings.

ATTRIBUTION: The majority of code in this file was generated with AI assistance.
I took advantage of AI to handle the tedious work of SEC API integration, HTML/XBRL parsing,
and regex-based section extraction from complex filing documents.
The core architecture and design decisions were mine.
"""
import requests
import time
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from datetime import datetime
import re


# SEC filing section patterns for extraction
SEC_SECTION_PATTERNS = {
    'item_1': r'item\s*1[.\s]+business',  # Business
    'item_1a': r'item\s*1a[.\s]+risk\s+factors',  # Risk Factors
    'item_1b': r'item\s*1b[.\s]',  # Unresolved Staff Comments
    'item_7': r'item\s*7[.\s]+management',  # MD&A
    'item_7a': r'item\s*7a[.\s]',  # Quantitative and Qualitative Disclosures
    'item_8': r'item\s*8[.\s]+financial',  # Financial Statements
}


class SECEdgarClient:
    """Client for interacting with SEC EDGAR API."""
    
    BASE_URL = "https://data.sec.gov"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions"
    
    def __init__(self, company_name: str = None, name: str = None, email: str = None, user_agent: str = None):
        """
        Initialize SEC EDGAR client.
        
        SEC requires User-Agent format: [Company Name] [Name] [Email]
        Example: "MyResearchApp JohnDoe john.doe@university.edu"
        
        Args:
            company_name: Your company/app name
            name: Your name
            email: Your email address
            user_agent: Pre-formatted user agent string (if provided, other params ignored)
        """
        if user_agent:
            self.user_agent = user_agent
        elif company_name and name and email:
            # Format: [Company Name] [Name] [Email]
            self.user_agent = f"{company_name} {name} {email}"
        else:
            # Fallback default
            self.user_agent = "InvestmentResearchApp User user@example.com"
            print("Warning: Using default User-Agent. Please configure SEC_EDGAR_COMPANY_NAME, SEC_EDGAR_NAME, and SEC_EDGAR_EMAIL in .env")
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Accept': 'application/json'
        })
    
    def get_company_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a company ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            CIK as string, or None if not found
        """
        try:
            # SEC provides a ticker to CIK mapping
            # Note: The endpoint is at www.sec.gov, not data.sec.gov
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            response = self.session.get(tickers_url, timeout=10)
            response.raise_for_status()
            
            tickers_data = response.json()
            
            # Search for the ticker
            for entry in tickers_data.values():
                if entry.get('ticker', '').upper() == ticker.upper():
                    cik = str(entry.get('cik_str', ''))
                    # Pad CIK to 10 digits
                    return cik.zfill(10)
            
            return None
        except Exception as e:
            print(f"Error fetching CIK for {ticker}: {e}")
            return None
    
    def get_company_filings(self, cik: str, form_type: str = "10-Q", limit: int = 4) -> List[Dict]:
        """
        Get recent filings for a company.
        
        Args:
            cik: Company CIK
            form_type: Type of filing (10-Q, 10-K, 8-K, etc.)
            limit: Maximum number of filings to return
            
        Returns:
            List of filing dictionaries with metadata
        """
        try:
            submissions_url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"
            response = self.session.get(submissions_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            filings = data.get('filings', {}).get('recent', {})
            
            # Filter by form type
            form_types = filings.get('form', [])
            filing_dates = filings.get('filingDate', [])
            accession_numbers = filings.get('accessionNumber', [])
            primary_documents = filings.get('primaryDocument', [])
            
            results = []
            for i, form in enumerate(form_types):
                if form == form_type and len(results) < limit:
                    results.append({
                        'form': form,
                        'filing_date': filing_dates[i],
                        'accession_number': accession_numbers[i],
                        'primary_document': primary_documents[i],
                        'cik': cik
                    })
            
            return results
        except Exception as e:
            print(f"Error fetching filings: {e}")
            return []
    
    def get_filing_content(self, cik: str, accession_number: str, document: str) -> Optional[str]:
        """
        Fetch and parse the content of a specific filing document.
        Handles modern Inline XBRL filings by properly extracting text.
        
        Args:
            cik: Company CIK
            accession_number: Filing accession number
            document: Document filename
            
        Returns:
            Parsed text content of the filing
        """
        try:
            # Remove dashes from accession number for the URL path
            accession_no_dash = accession_number.replace('-', '')
            url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_no_dash}/{document}"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Use BeautifulSoup to properly parse Inline XBRL
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove XBRL hidden data sections (they contain duplicated/machine data)
            for tag in soup(['script', 'style', 'ix:header', 'ix:hidden']):
                tag.decompose()
            
            # Extract text while preserving structure
            text_content = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            text_content = re.sub(r'\s+', ' ', text_content)
            text_content = re.sub(r' ([.,;:!?])', r'\1', text_content)  # Fix punctuation spacing
            
            return text_content
        except Exception as e:
            print(f"Error fetching filing content: {e}")
            return None
    
    def get_quarterly_filings(self, ticker: str, limit: int = 4) -> List[Dict]:
        """
        Get latest quarterly filings (10-Q) for a company.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of filings to return
            
        Returns:
            List of filing dictionaries with content
        """
        return self.get_filings(ticker, form_type="10-Q", limit=limit)
    
    def get_annual_filings(self, ticker: str, limit: int = 2) -> List[Dict]:
        """
        Get latest annual filings (10-K) for a company.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of filings to return
            
        Returns:
            List of filing dictionaries with content
        """
        return self.get_filings(ticker, form_type="10-K", limit=limit)
    
    def get_material_event_filings(self, ticker: str, limit: int = 5) -> List[Dict]:
        """
        Get latest material event filings (8-K) for a company.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of filings to return
            
        Returns:
            List of filing dictionaries with content
        """
        return self.get_filings(ticker, form_type="8-K", limit=limit)
    
    def get_filings(self, ticker: str, form_type: str = "10-Q", limit: int = 4) -> List[Dict]:
        """
        Get filings of a specific type for a company.
        
        Args:
            ticker: Stock ticker symbol
            form_type: Type of filing (10-Q, 10-K, 8-K)
            limit: Maximum number of filings to return
            
        Returns:
            List of filing dictionaries with content
        """
        cik = self.get_company_cik(ticker)
        if not cik:
            return []
        
        filings = self.get_company_filings(cik, form_type=form_type, limit=limit)
        
        # Fetch content for each filing
        results = []
        for filing in filings:
            content = self.get_filing_content(
                filing['cik'],
                filing['accession_number'],
                filing['primary_document']
            )
            
            if content:
                filing['content'] = content
                filing['ticker'] = ticker
                filing['url'] = f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={filing['accession_number']}&xbrl_type=v"
                results.append(filing)
            
            # Be respectful with rate limiting
            time.sleep(0.1)
        
        return results
    
    def extract_section(self, content: str, section: str) -> Optional[str]:
        """
        Extract a specific section from a filing, skipping the Table of Contents.

        Args:
            content: Full filing content
            section: Section to extract ('item_1', 'item_1a', 'item_7', 'item_8')

        Returns:
            Extracted section text or None
        """
        if section not in SEC_SECTION_PATTERNS:
            return None

        pattern = SEC_SECTION_PATTERNS[section]
        content_lower = content.lower()

        # Strategy: Find PART I or PART II to skip TOC, then find the section
        # Most 10-Ks have: TOC -> Forward-looking statements -> PART I -> Item 1

        # Find where actual content starts (after TOC)
        # Look for "PART I" or "PART II" markers
        part_markers = ['part i\n', 'part i ', 'part ii\n', 'part ii ']
        content_start = 0

        for marker in part_markers:
            part_match = content_lower.find(marker)
            if part_match > 0 and part_match < 100000:  # Reasonable position
                content_start = part_match
                break

        # If no PART marker found, skip first 30K chars (usually TOC + XBRL)
        if content_start == 0:
            content_start = min(30000, len(content) // 4)

        # Now search for the section AFTER the TOC
        search_region = content_lower[content_start:]
        match = re.search(pattern, search_region)

        if not match:
            # Fallback: try the whole document
            match = re.search(pattern, content_lower)
            if not match:
                return None
            start_idx = match.start()
        else:
            start_idx = content_start + match.start()

        # Find next section (approximate end)
        # Look for next "Item X" pattern after current match
        search_after = content_lower[start_idx + 100:]  # Skip current header
        next_section = re.search(r'item\s*\d+[a-z]?[.\s]', search_after)

        if next_section:
            end_idx = start_idx + 100 + next_section.start()
        else:
            # Take a reasonable chunk if no next section found
            end_idx = min(start_idx + 50000, len(content))

        extracted = content[start_idx:end_idx].strip()

        # Sanity check: if extracted section is very short, it might be TOC
        if len(extracted) < 500:
            # Try finding the next occurrence
            search_region2 = content_lower[start_idx + 200:]
            match2 = re.search(pattern, search_region2)
            if match2:
                start_idx2 = start_idx + 200 + match2.start()
                next_section2 = re.search(r'item\s*\d+[a-z]?[.\s]', content_lower[start_idx2 + 100:])
                if next_section2:
                    end_idx2 = start_idx2 + 100 + next_section2.start()
                else:
                    end_idx2 = min(start_idx2 + 50000, len(content))
                extracted = content[start_idx2:end_idx2].strip()

        return extracted
    
    def get_filings_with_sections(self, ticker: str, form_type: str = "10-K", 
                                   sections: List[str] = None, limit: int = 1) -> List[Dict]:
        """
        Get filings with specific sections extracted.
        
        Args:
            ticker: Stock ticker symbol
            form_type: Type of filing (10-K, 10-Q)
            sections: List of sections to extract (e.g., ['item_1', 'item_1a', 'item_7'])
            limit: Maximum number of filings
            
        Returns:
            List of filing dictionaries with extracted sections
        """
        if sections is None:
            sections = ['item_1', 'item_1a', 'item_7']
        
        filings = self.get_filings(ticker, form_type=form_type, limit=limit)
        
        for filing in filings:
            content = filing.get('content', '')
            filing['sections'] = {}
            
            for section in sections:
                extracted = self.extract_section(content, section)
                if extracted:
                    filing['sections'][section] = extracted
        
        return filings

