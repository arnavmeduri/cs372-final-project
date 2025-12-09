"""
Section Validation and Fallback Strategies

Validates quality of extracted SEC filing sections and provides fallback
strategies when primary extraction fails.


ATTRIBUTION: The majority of code in this file was generated with AI assistance.
I took advantage of AI to deal with various edge cases in validation heuristics.
"""

import re
from typing import Dict, Tuple, Optional


class SectionValidator:
    """
    Validates extracted section quality and provides fallback extraction.
    """


    MIN_LENGTH_NARRATIVE = 500  
    MIN_LENGTH_SHORT = 200     


    MIN_ALPHA_RATIO = 0.5

    def __init__(self, verbose: bool = True):
        """
        Initialize section validator.

        Args:
            verbose: Print validation warnings
        """
        self.verbose = verbose

    def validate_section_quality(
        self,
        section_content: str,
        section_name: str,
        is_narrative: bool = True
    ) -> Tuple[bool, str]:
        """
        Check if extracted section has sufficient content.

        Args:
            section_content: Extracted section text
            section_name: Name of section (for logging)
            is_narrative: True if narrative section (Business, MD&A, Risks)

        Returns:
            (is_valid, failure_reason)
        """
        if not section_content:
            reason = f"{section_name}: Empty content"
            if self.verbose:
                print(f"warning: {reason}")
            return False, reason


        min_length = self.MIN_LENGTH_NARRATIVE if is_narrative else self.MIN_LENGTH_SHORT
        if len(section_content) < min_length:
            reason = f"{section_name}: Only {len(section_content)} chars (expected >{min_length})"
            if self.verbose:
                print(f"warning: {reason}")
            return False, reason

        # Check alphabetic ratio (not mostly numbers/tables)
        alpha_count = sum(c.isalpha() for c in section_content)
        alpha_ratio = alpha_count / len(section_content) if len(section_content) > 0 else 0

        if alpha_ratio < self.MIN_ALPHA_RATIO:
            reason = f"{section_name}: Only {alpha_ratio:.0%} alphabetic (expected >{self.MIN_ALPHA_RATIO:.0%})"
            if self.verbose:
                print(f"warning: {reason}")
            return False, reason

        # Check for common extraction errors
        error_indicators = [
            'table of contents',
            '───',  # Table separators
            'page ',  # Page numbers only
        ]

        # Check first 500 chars for error indicators
        sample = section_content[:500].lower()
        for indicator in error_indicators:
            if indicator in sample:
                reason = f"{section_name}: Contains '{indicator}' in first 500 chars (likely extraction error)"
                if self.verbose:
                    print(f"warning: {reason}")
                return False, reason

        # Passed all checks
        if self.verbose:
            print(f"{section_name}: valid ({len(section_content):,} chars, {alpha_ratio:.0%} alphabetic)")
        return True, ""

    def extract_forward_looking_statements(self, content: str, max_length: int = 5000) -> str:
        """
        Extract forward-looking statements from content.

        Useful fallback when MD&A section fails but Business section contains
        strategy/outlook information.

        Args:
            content: Source content (usually Business section)
            max_length: Maximum chars to extract

        Returns:
            Extracted forward-looking content
        """
        # Keywords that indicate forward-looking/strategy content
        forward_keywords = [
            'strategy', 'strategic', 'future', 'outlook', 'expect', 'plan',
            'growth', 'expansion', 'opportunity', 'opportunities',
            'initiative', 'initiatives', 'investment', 'focus',
            'believe', 'anticipate', 'intend', 'continue to',
            'will ', 'may ', 'could ', 'should '
        ]

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)

        # Score each paragraph by forward-looking keyword density
        scored_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) < 100:  # Skip very short paragraphs
                continue

            # Count forward-looking keywords
            para_lower = para.lower()
            score = sum(1 for kw in forward_keywords if kw in para_lower)

            if score > 0:
                scored_paragraphs.append((score, para))

        # Sort by score (most forward-looking first)
        scored_paragraphs.sort(reverse=True, key=lambda x: x[0])

        # Collect top paragraphs up to max_length
        extracted = []
        total_length = 0

        for score, para in scored_paragraphs:
            if total_length + len(para) > max_length:
                break
            extracted.append(para)
            total_length += len(para)

        result = '\n\n'.join(extracted)

        if self.verbose and result:
            print(f"extracted {len(result):,} chars of forward-looking content from fallback")

        return result

    def extract_risk_keywords(self, content: str, max_length: int = 8000) -> str:
        """
        Extract risk-related content from full filing.

        Fallback when Risk Factors section fails.

        Args:
            content: Full filing content
            max_length: Maximum chars to extract

        Returns:
            Extracted risk content
        """
        # Risk-related keywords
        risk_keywords = [
            'risk', 'risks', 'could adversely', 'may adversely', 'may negatively',
            'uncertainty', 'uncertainties', 'challenge', 'challenges',
            'threat', 'threats', 'failure', 'decline', 'loss',
            'disruption', 'volatile', 'volatility', 'competition',
            'regulatory', 'litigation', 'compliance'
        ]

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)

        # Score sentences by risk keyword density
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 50:  # Skip very short sentences
                continue

            sentence_lower = sentence.lower()
            score = sum(1 for kw in risk_keywords if kw in sentence_lower)

            if score > 0:
                scored_sentences.append((score, sentence))

        # Sort by score
        scored_sentences.sort(reverse=True, key=lambda x: x[0])

        # Collect top sentences up to max_length
        extracted = []
        total_length = 0

        for score, sentence in scored_sentences[:100]:  # Top 100 risk sentences
            if total_length + len(sentence) > max_length:
                break
            extracted.append(sentence)
            total_length += len(sentence)

        result = ' '.join(extracted)

        if self.verbose and result:
            print(f"extracted {len(result):,} chars of risk content from fallback")

        return result

    def get_section_with_fallback(
        self,
        doc,
        section_name: str,
        company_name: str,
        verbose: bool = True
    ) -> Tuple[str, str]:
        """
        Extract section with fallback strategies for robustness.

        Args:
            doc: edgartools document object (TenK, TenQ, etc.)
            section_name: Section to extract ('business', 'risk_factors', 'management_discussion')
            company_name: Company name (for logging)
            verbose: Print status messages

        Returns:
            (content, status_message)
            status can be: 'primary', 'fallback', 'failed'
        """
        # Map section names to edgartools attributes
        attr_mapping = {
            'business': 'business',
            'risk_factors': 'risk_factors',
            'management_discussion': 'management_discussion'
        }

        if section_name not in attr_mapping:
            return "", f"Unknown section: {section_name}"

        # Try primary extraction
        attr_name = attr_mapping[section_name]
        primary_content = getattr(doc, attr_name, "")

        # Validate primary extraction
        is_valid, reason = self.validate_section_quality(
            primary_content,
            section_name,
            is_narrative=True
        )

        if is_valid:
            return primary_content, "primary"

        # Primary extraction failed - try fallbacks
        if verbose:
            print(f"\n[FALLBACK] {section_name} validation failed: {reason}")
            print(f"[FALLBACK] Attempting alternative extraction...")

        # Fallback strategies by section type
        if section_name == 'management_discussion':
            # MD&A fallback: Extract forward-looking statements from Business section
            business_content = getattr(doc, 'business', "")
            if business_content:
                fallback_content = self.extract_forward_looking_statements(business_content)
                if len(fallback_content) > 500:
                    if verbose:
                        print(f"[fallback] successfully extracted {len(fallback_content):,} chars from business section")
                    return fallback_content, "fallback_business"

        elif section_name == 'risk_factors':
            # Risk fallback: Search full filing for risk content
            try:
                full_text = doc.text() if hasattr(doc, 'text') else ""
                if full_text:
                    fallback_content = self.extract_risk_keywords(full_text)
                    if len(fallback_content) > 500:
                        if verbose:
                            print(f"[fallback] successfully extracted {len(fallback_content):,} chars from full filing")
                        return fallback_content, "fallback_full_filing"
            except Exception as e:
                if verbose:
                    print(f"warning [fallback] failed to extract from full filing: {e}")

        elif section_name == 'business':
            # Business fallback: Try to get company description from CIK lookup
            # (Could be implemented if needed)
            pass

        # All fallbacks failed
        if verbose:
            print(f"warning [fallback] all fallbacks failed for {section_name}")
            print(f"warning: {section_name} quality low, results may be limited")

        # Return original content with warning
        return primary_content, "failed"
