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
from src.rich_formatter import RichAnalysisFormatter

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
    if "metrics" not in st.session_state:
        st.session_state.metrics = None
    
    # Generate report when button clicked
    if generate_clicked and ticker:
        st.session_state.report = None
        st.session_state.metrics = None
        
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
                
                # Generate the report
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
            
            # Investment Score Section - Simple display
            from src.investment_score import calculate_investment_score
            score_data = calculate_investment_score(metrics)
            
            st.markdown("### Investment Score")
            
            col_score1, col_score2 = st.columns(2)
            with col_score1:
                st.metric("Score", f"{score_data['score']}/100")
            with col_score2:
                rec = score_data['recommendation']
                if rec == "BUY":
                    st.markdown(f"**Signal:** :green[{rec}]")
                elif rec == "HOLD":
                    st.markdown(f"**Signal:** :orange[{rec}]")
                else:
                    st.markdown(f"**Signal:** :red[{rec}]")
            
            st.markdown(f"*{score_data.get('methodology', 'Graham Defensive Investor Score')}*", unsafe_allow_html=True)
            if score_data.get('criteria_met') and score_data.get('total_criteria'):
                st.markdown(f"<p style='font-size:0.75rem; color:#6c757d;'>Meets {score_data['criteria_met']} of {score_data['total_criteria']} measurable Graham criteria</p>", unsafe_allow_html=True)
            st.markdown("")
        
        # Main Analysis Content (company_summary contains the full rich analysis)
        if brief.company_summary:
            st.markdown("## Analysis")
            # Clean up any problematic markdown formatting
            analysis_text = brief.company_summary
            # Remove any leading/trailing code block markers that cause green text
            analysis_text = analysis_text.strip()
            if analysis_text.startswith("```"):
                # Find and remove opening code block
                first_newline = analysis_text.find('\n')
                if first_newline != -1:
                    analysis_text = analysis_text[first_newline+1:]
            if analysis_text.endswith("```"):
                analysis_text = analysis_text[:-3]
            analysis_text = analysis_text.strip()
            st.markdown(analysis_text)
        
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

