"""
FinBrief Streamlit Web Application
Clean, professional UI for generating financial research reports.
"""
import streamlit as st
import sys
import os

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.finbrief import FinBriefApp
from src.utils.rich_formatter import RichAnalysisFormatter

# Page configuration
st.set_page_config(
    page_title="FinBrief - Financial Research",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional look (works in both light and dark mode)
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    /* Sidebar styling - works in both modes */
    [data-testid="stSidebar"] {
        padding-top: 1rem;
    }

    /* Headers - visible in both light and dark mode */
    h1 {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    h2 {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    h3 {
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Dividers */
    hr {
        margin: 1rem 0;
        border: none;
        opacity: 0.3;
    }

    /* Metric cards - work in both modes */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 600;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        opacity: 0.7;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        padding: 0.6rem 1rem;
        font-weight: 500;
    }

    /* Status text */
    .status-text {
        font-size: 0.85rem;
        opacity: 0.7;
    }

    /* Sources section */
    .sources {
        font-size: 0.85rem;
        line-height: 1.6;
        opacity: 0.8;
    }

    /* About section in sidebar */
    .about-text {
        font-size: 0.85rem;
        line-height: 1.5;
        opacity: 0.8;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Spinner text */
    .stSpinner > div > div {
        color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)


def format_market_cap(value):
    """Format market cap value."""
    if value is None:
        return "N/A"
    if value >= 1e6:
        return f"${value/1e6:.2f}T"
    elif value >= 1e3:
        return f"${value/1e3:.2f}B"
    else:
        return f"${value:.2f}M"


def format_metric(value, suffix="", prefix="", decimals=2):
    """Format a metric value."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        formatted = f"{value:.{decimals}f}"
        if value > 0 and prefix == "+/-":
            prefix = "+"
        elif value < 0 and prefix == "+/-":
            prefix = ""
        return f"{prefix}{formatted}{suffix}"
    return str(value)


def clean_markdown_text(text: str) -> str:
    """
    Clean LLM-generated markdown to prevent rendering issues in Streamlit.
    
    Removes:
    - Code blocks (```)
    - Inline code backticks (`)
    - LaTeX math delimiters ($, $$)
    - Streamlit color syntax (:color[...])
    """
    import re
    
    cleaned = text.strip()
    
    # Remove opening code blocks (with or without language specifier)
    if cleaned.startswith("```"):
        first_newline = cleaned.find('\n')
        if first_newline != -1:
            cleaned = cleaned[first_newline+1:]
    
    # Remove closing code blocks
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    # Remove mid-text code blocks
    cleaned = re.sub(r'```\w*\n', '', cleaned)  # Opening blocks
    cleaned = re.sub(r'\n```', '', cleaned)     # Closing blocks
    
    # Remove inline code backticks (causes green text in Streamlit)
    # Only remove single backticks, not triple backticks
    cleaned = re.sub(r'(?<!`)`([^`\n]+)`(?!`)', r'\1', cleaned)
    
    # Remove LaTeX-style math delimiters (can cause character-by-character rendering)
    cleaned = re.sub(r'\$\$([^$]+)\$\$', r'\1', cleaned)  # Display math
    cleaned = re.sub(r'\$([^$\n]+)\$', r'\1', cleaned)     # Inline math
    
    # Remove Streamlit color syntax
    cleaned = re.sub(r':\w+\[([^\]]+)\]', r'\1', cleaned)
    
    return cleaned.strip()


def main():
    # ─────────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ─────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### FinBrief: Instructional Financial Reports")
        
        st.markdown("---")
        
        # Ticker input
        ticker = st.text_input(
            "Stock Ticker",
            value="",
            placeholder="e.g., AAPL, MSFT, IBM",
            help="Enter a valid stock ticker symbol"
        ).upper().strip()
        
        # Filing type
        filing_type = st.radio(
            "Filing Type",
            options=["10-K", "10-Q"],
            index=0,
            help="10-K: Annual report | 10-Q: Quarterly report"
        )

        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            # Comparison mode
            compare_mode = st.checkbox(
                "Generate Both (RAG vs No-RAG)",
                value=False,
                help="Generate two reports side-by-side to compare RAG impact. Shows how RAG improves quality with SEC filings vs pure LLM knowledge."
            )

            if compare_mode:
                st.info("Comparison mode: Will generate both WITH RAG and WITHOUT RAG reports side-by-side")

        st.markdown("---")

        # Generate button
        generate_clicked = st.button(
            "Generate Report",
            type="primary"
        )
        
        st.markdown("---")
        
        # About section
        st.markdown("**About**")
        st.markdown(
            """<div class="about-text">
            FinBrief is an AI-powered research tool that generates 
            financial analysis reports using SEC filings and real-time 
            market data. Reports are for educational purposes only.
            </div>""",
            unsafe_allow_html=True
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # MAIN CONTENT
    # ─────────────────────────────────────────────────────────────────────────
    
    # Initialize session state for report
    if "report" not in st.session_state:
        st.session_state.report = None
    if "report_no_rag" not in st.session_state:
        st.session_state.report_no_rag = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = None
    if "is_comparison" not in st.session_state:
        st.session_state.is_comparison = False

    # Generate report when button clicked
    if generate_clicked and ticker:
        st.session_state.report = None
        st.session_state.report_no_rag = None
        st.session_state.metrics = None
        st.session_state.is_comparison = compare_mode

        if compare_mode:
            # Comparison mode: generate both reports
            with st.spinner(f"Generating comparison for {ticker}... This may take 60-120 seconds."):
                try:
                    # Initialize FinBrief
                    app = FinBriefApp(
                        use_duke_gateway=True,
                        verbose=False
                    )

                    # Fetch metrics first (for display)
                    metrics = app.fetch_metrics(ticker)
                    st.session_state.metrics = metrics

                    # Generate WITH RAG
                    st.info("[1/2] Generating WITH RAG (SEC filings + financial data)...")
                    brief_rag = app.generate_rich_analysis(
                        ticker=ticker,
                        filing_type=filing_type,
                        use_rag=True
                    )
                    st.session_state.report = brief_rag

                    # Generate WITHOUT RAG
                    st.info("[2/2] Generating WITHOUT RAG (LLM general knowledge only)...")
                    brief_no_rag = app.generate_rich_analysis(
                        ticker=ticker,
                        filing_type=filing_type,
                        use_rag=False
                    )
                    st.session_state.report_no_rag = brief_no_rag

                    st.success("✅ Both reports generated successfully!")

                except Exception as e:
                    st.error(f"Error generating comparison: {str(e)}")
                    st.session_state.report = None
                    st.session_state.report_no_rag = None
        else:
            # Standard mode: single report WITH RAG (default)
            with st.spinner(f"Analyzing {ticker}... This may take 30-60 seconds."):
                try:
                    # Initialize FinBrief
                    app = FinBriefApp(
                        use_duke_gateway=True,
                        verbose=False
                    )

                    # Fetch metrics first (for display)
                    metrics = app.fetch_metrics(ticker)
                    st.session_state.metrics = metrics

                    # Generate the report WITH RAG
                    brief = app.generate_rich_analysis(
                        ticker=ticker,
                        filing_type=filing_type,
                        use_rag=True
                    )

                    st.session_state.report = brief

                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    st.session_state.report = None
    
    # Display report if available
    if st.session_state.report:
        # Comparison mode: Display both reports side-by-side
        if st.session_state.is_comparison and st.session_state.report_no_rag:
            brief_rag = st.session_state.report
            brief_no_rag = st.session_state.report_no_rag
            metrics = st.session_state.metrics

            # Header
            st.markdown(f"# RAG vs No-RAG Comparison: {brief_rag.company_name} ({brief_rag.ticker})")
            st.markdown(f"*Analysis Date: {brief_rag.analysis_date}*")

            st.markdown("---")

            st.info("""
            **About this comparison:** This demonstrates the qualitative impact of Retrieval-Augmented Generation (RAG).
            The left report uses actual SEC filings and real-time financial data, while the right report uses only the LLM's general knowledge.
            """)

            st.markdown("---")

            # Key Metrics Section (shared for both RAG and No-RAG)
            if metrics:
                st.markdown("## Key Metrics")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Market Cap", format_market_cap(metrics.market_cap))
                with col2:
                    st.metric("P/E Ratio", format_metric(metrics.pe_ratio))
                with col3:
                    st.metric("EPS (TTM)", format_metric(metrics.eps_ttm, prefix="$"))

                col4, col5, col6 = st.columns(3)
                with col4:
                    st.metric("Revenue Growth", format_metric(metrics.revenue_growth_yoy, suffix="%", prefix="+/-"))
                with col5:
                    st.metric("Debt/Equity", format_metric(metrics.debt_to_equity))
                with col6:
                    st.metric("Dividend Yield", format_metric(metrics.dividend_yield, suffix="%"))

                st.markdown("")

            # Sentiment Analysis Section (only for RAG report)
            if brief_rag.sentiment_analysis:
                st.markdown("## Filing Sentiment")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive", f"{brief_rag.sentiment_analysis.positive_pct:.1f}%")
                with col2:
                    st.metric("Neutral", f"{brief_rag.sentiment_analysis.neutral_pct:.1f}%")
                with col3:
                    st.metric("Negative", f"{brief_rag.sentiment_analysis.negative_pct:.1f}%")
                st.markdown(f"*Overall Tone: {brief_rag.sentiment_analysis.overall_tone}*")
                st.markdown("")

            st.markdown("---")
            st.markdown("## Analysis Comparison")

            # Side-by-side comparison using columns
            col_rag, col_no_rag = st.columns(2)

            with col_rag:
                st.markdown("### WITH RAG")
                st.markdown("")

                # Display RAG analysis
                if brief_rag.company_summary:
                    analysis_text = clean_markdown_text(brief_rag.company_summary)
                    st.markdown(analysis_text, unsafe_allow_html=False)

                st.markdown("")

                # Sources
                if brief_rag.summary_citations:
                    st.markdown("**Sources:**")
                    for citation in brief_rag.summary_citations:
                        lines = citation.split('\n')
                        main_ref = lines[0]
                        url = None
                        for line in lines[1:]:
                            if line.strip().startswith("URL:"):
                                url = line.strip().replace("URL:", "").strip()
                                break
                        if url:
                            st.markdown(f"🔗 [{main_ref}]({url})")
                        else:
                            st.markdown(f"- {main_ref}")

            with col_no_rag:
                st.markdown("### WITHOUT RAG")
                st.markdown("")

                # Display No-RAG analysis
                if brief_no_rag.company_summary:
                    analysis_text = clean_markdown_text(brief_no_rag.company_summary)
                    st.markdown(analysis_text, unsafe_allow_html=False)

                st.markdown("")
                st.warning("⚠️ No source citations - based only on LLM general knowledge")

            return  # Exit early, don't display single report

        # Standard mode: Display single report
        brief = st.session_state.report
        metrics = st.session_state.metrics

        # Header
        st.markdown(f"# {brief.company_name} ({brief.ticker})")
        st.markdown(f"*Analysis Date: {brief.analysis_date}*")
        
        st.markdown("---")
        
        # Key Metrics Section
        if metrics:
            st.markdown("## Key Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Cap", format_market_cap(metrics.market_cap))
            with col2:
                st.metric("P/E Ratio", format_metric(metrics.pe_ratio))
            with col3:
                st.metric("EPS (TTM)", format_metric(metrics.eps_ttm, prefix="$"))
            
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Revenue Growth", format_metric(metrics.revenue_growth_yoy, suffix="%", prefix="+/-"))
            with col5:
                st.metric("Debt/Equity", format_metric(metrics.debt_to_equity))
            with col6:
                st.metric("Dividend Yield", format_metric(metrics.dividend_yield, suffix="%"))
            
            st.markdown("")

        # Sentiment Analysis Section
        if brief.sentiment_analysis:
            st.markdown("## Filing Sentiment")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positive", f"{brief.sentiment_analysis.positive_pct:.1f}%")
            with col2:
                st.metric("Neutral", f"{brief.sentiment_analysis.neutral_pct:.1f}%")
            with col3:
                st.metric("Negative", f"{brief.sentiment_analysis.negative_pct:.1f}%")
            st.markdown(f"*Overall Tone: {brief.sentiment_analysis.overall_tone}*")
            st.markdown("")
        
        # Main Analysis Content (company_summary contains the full rich analysis)
        if brief.company_summary:
            st.markdown("## Analysis")
            # Clean up any problematic markdown formatting
            analysis_text = clean_markdown_text(brief.company_summary)
            st.markdown(analysis_text, unsafe_allow_html=False)
        
        st.markdown("---")
        
        # Sources - Links to SEC filings (one per filing type)
        if brief.summary_citations:
            st.markdown("## Sources")
            st.markdown("*All statements in this report are grounded in the following SEC filings:*")
            st.markdown("")
            for citation in brief.summary_citations:
                # Split citation into main reference and URL
                lines = citation.split('\n')
                main_ref = lines[0]
                
                # Check if there's a URL
                url = None
                for line in lines[1:]:
                    if line.strip().startswith("URL:"):
                        url = line.strip().replace("URL:", "").strip()
                        break
                
                if url:
                    # Display as clickable link
                    st.markdown(f"**{main_ref}**")
                    st.markdown(f"🔗 [View on SEC.gov]({url})")
                else:
                    # No URL, just display the citation
                    st.markdown(f"**{main_ref}**")
                st.markdown("")
        
        st.markdown("")
        
        # Download button
        formatter = RichAnalysisFormatter()
        report_text = formatter.format_text(brief)
        
        st.download_button(
            label="Download Report",
            data=report_text,
            file_name=f"{brief.ticker}_report.txt",
            mime="text/plain"
        )
        
        # Disclaimer
        st.markdown("---")
        st.markdown(
            """<div class="sources">
            <strong>Disclaimer:</strong> This report is for educational purposes only 
            and should not be considered investment advice. Always conduct your own 
            research before making investment decisions.
            </div>""",
            unsafe_allow_html=True
        )
    
    else:
        # Welcome message when no report
        st.markdown("# FinBrief: Instructional Financial Reports")
        
        st.markdown("---")
        
        st.markdown("""
        Generate comprehensive financial reports powered by RAG 
        (Retrieval-Augmented Generation) using SEC filings and real-time market data.
        
        **Steps:**
        1. Enter a stock ticker in the sidebar on the left. You can find tickers for publicly-traded companies at [here](https://stockanalysis.com/stocks/).
        2. Select filing type (10-K for annual, 10-Q for quarterly)
        3. Click the "Generate Report" button to produce a report
        4. Review your generated report (and download if you want to save it)!
        """)
        
        st.markdown("---")
        
        st.markdown(
            """<div class="sources">
            Reports typically take 30-60 seconds to generate. 
            The analysis includes data from SEC EDGAR filings and Finnhub financial APIs.
            </div>""",
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()

