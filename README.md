# FinBrief: Educational Financial Briefs for Beginning Investors

An AI-powered educational tool that generates structured, citation-grounded financial briefs for students learning about investing. FinBrief uses Retrieval-Augmented Generation (RAG) to extract information from SEC filings and financial data, then presents it in a beginner-friendly format.

## What it Does

FinBrief helps beginning investors understand publicly traded companies by:

- **Retrieving authoritative SEC filings** (10-K, 10-Q, 8-K) directly from SEC EDGAR using edgartools
- **Validating section quality** with automatic fallback strategies when extraction fails (Phase 3A)
- **Analyzing balance sheets** with 5 key financial ratios (Current Ratio, Quick Ratio, D/E, Working Capital, Cash Ratio) (Phase 3A)
- **Fetching real-time financial metrics** from Finnhub (P/E ratio, EPS, Market Cap, etc.)
- **Generating structured educational briefs** with grounded, hallucination-free content using GPT 4.1
- **Providing transparent citations** for every factual claim with source attribution

**This is an educational tool, NOT investment advice.**

## Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd CS372

# 2. Run the setup script
./setup_venv.sh

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Activate virtual environment
source venv/bin/activate

# 5. Run the Web UI (recommended)
streamlit run app.py

# 6. Or use command line
python -m src.finbrief AAPL --duke-gateway

# 7. Test with any company ticker
python data/test_any_company.py MSFT
```

### Output Modes

**Standard Mode (Quick Analysis)**
```bash
python -m src.finbrief AAPL
```
- Comprehensive analysis with RAG
- ~30-60 seconds

**Rich Mode (Detailed Research)**
```bash
python -m src.finbrief AAPL --duke-gateway
```
- Detailed research-quality analysis
- Duke AI Gateway (GPT 4.1)
- Best quality output

See `docs/QUICK_CLI_REFERENCE.md` for more details.

## Video Links

- **Demo Video:** [Link to demo video]
- **Technical Walkthrough:** [Link to technical walkthrough]

## Evaluation

### Quantitative Results

| Metric | Value |
|--------|-------|
| SEC Filing Retrieval Success | 100% (10-K, 10-Q, 8-K support) |
| Embedding Model | all-MiniLM-L6-v2 (384-dim) |
| RAG Retrieval Accuracy | Semantic similarity via FAISS |
| Confidence Score Range | 0.30 - 0.95 (heuristic) |

### Model Architecture

| Component | Model | Parameters |
|-----------|-------|------------|
| Embedding | all-MiniLM-L6-v2 | 22M |
| Generator (Local) | TinyLlama-1.1B-Chat | 1.1B |
| Generator (Gateway) | GPT 4.1 (Duke AI) | Frontier |
| LoRA Adapter | r=16, α=32 | 4.3M (1.2%) |

### Sample Output Sections

The FinBrief output includes 10 structured sections:

1. **Company Summary** - Beginner-friendly business description
2. **Key Financial Metrics** - P/E, EPS, Market Cap with interpretations
3. **Student Interpretation** - What the metrics mean for beginners
4. **Risks** - From SEC filings with severity ratings
5. **Opportunities** - Growth potential with categories
6. **What This Means** - Actionable takeaways for new investors
7. **Key Terms Explained** - Financial vocabulary definitions
8. **Beginner Difficulty Score** - Easy/Intermediate/Advanced rating
9. **Educational Summary** - Learning-focused overview
10. **Confidence Score** - Model certainty estimate

### Test Results

```
48 passed, 5 skipped (network tests)
```

## Individual Contributions

| Team Member | Contributions |
|-------------|---------------|
| Arnav Meduri | Solo project - all components |

## Project Structure

```
CS372/
├── README.md                 # Project overview
├── ATTRIBUTION.md            # AI and resource attribution
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── setup_venv.sh             # Setup script (Unix/Mac)
├── setup_venv.ps1            # Setup script (Windows)
│
├── src/                      # Source code
│   ├── finbrief.py           # Main FinBrief application
│   ├── sec_edgar_client.py   # SEC EDGAR API client (legacy)
│   ├── edgartools_client.py  # edgartools wrapper (Phase 2)
│   ├── section_validator.py  # Section quality validation (Phase 3A)
│   ├── balance_sheet_analyzer.py  # Balance sheet analysis (Phase 3A)
│   ├── finnhub_client.py     # Finnhub metrics client
│   ├── rag_system.py         # RAG with FAISS
│   ├── model_handler.py      # Local model handler + LoRA
│   ├── duke_gateway_model.py # Duke AI Gateway integration
│   ├── educational_brief.py  # Structured output format
│   ├── rich_formatter.py     # Rich analysis formatter
│   ├── confidence_head.py    # Confidence estimation
│   ├── prompt_loader.py      # Prompt management
│   ├── lora_trainer.py       # LoRA fine-tuning
│   ├── training_data_builder.py  # Training data generation
│   └── finetune.py           # Fine-tuning CLI
│
├── data/                     # Data and data access scripts
│   ├── training/             # Training data for LoRA
│   │   └── finbrief_training.json
│   ├── test_finnhub.py       # Finnhub API test script
│   ├── test_duke_gateway.py  # Duke Gateway test script
│   └── test_any_company.py   # General company analysis test
│
├── models/                   # Trained models and configurations
│   └── lora_adapter/         # Fine-tuned LoRA weights
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       └── training_metrics.json
│
├── notebooks/                # Jupyter notebooks (if any)
│
├── docs/                     # Documentation
│   ├── QUICK_CLI_REFERENCE.md  # CLI usage guide
│   └── prompts.md            # LLM prompts configuration
│
├── tests/                    # Unit tests
│   ├── test_finnhub_client.py
│   ├── test_rag_system.py
│   └── test_sec_edgar_client.py
│
└── videos/                   # Demo and walkthrough videos
```

## Key Features

### Phase 3A: Enhanced Section Processing (Latest)
- **Section Validation**: Automatically detects extraction failures (< 500 chars, low alphabetic ratio)
- **Fallback Strategies**: Extracts forward-looking statements from Business section when MD&A fails
- **Balance Sheet Analysis**: Calculates 5 key ratios with beginner-friendly interpretations:
  - Current Ratio (liquidity)
  - Quick Ratio (immediate liquidity)
  - Debt-to-Equity (leverage)
  - Working Capital (operational health)
  - Cash Ratio (cash position)
- **Hallucination Elimination**: 0% generic content (validated across Tech, Finance, Manufacturing, Healthcare)
- **Universal Generalization**: Works for any company regardless of industry

### RAG Pipeline
- **Embedding**: all-MiniLM-L6-v2 (SentenceTransformers)
- **Vector Store**: FAISS for efficient similarity search
- **Sources**: SEC filings (edgartools) + Finnhub metrics

### Model
- **Base**: GPT-2 Medium (355M parameters)
- **Fine-tuning**: LoRA (r=16, α=32, 1.2% trainable)
- **Production**: Duke AI Gateway (GPT 4.1)
- **Alternatives**: TinyLlama supported for local inference

### Data Sources
- **SEC EDGAR**: Official filings (10-K, 10-Q) via edgartools
- **Finnhub**: Real-time financial metrics
- **Duke AI Gateway**: GPT 4.1 for analysis

## License

This project is for educational and research purposes only.
