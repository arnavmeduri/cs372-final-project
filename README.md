# CS372 Final Project: FinBrief - Educational Financial Analysis
Arnav Meduri

## What It Does

My project is an educational financial analysis tool that generates beginner-friendly investment reports for publicly traded companies. The goal of this project was to help beginner investors and students understand key aspects of publicly traded companies without getting overwhelmed by complex financial jargon/termiology. The system retrieves authoritative information from SEC filings, mandatory financial and business reports that public and private companies submit to the US Securities and Exchange Commission (via the EDGAR database), and up-to-date financial metrics from Finnhub, and then performs sentiment analysis on the retrieved filing chunks using a fine-tuned DistilBERT model. The RAG system makes use of semantic embeddings (all-MiniLM-L6-v2) and FAISS for efficient similarity search using L2-normalized cosine similarity. Reports are generated using Duke AI Gateway, which provides access to frontier models like GPT-4.1 and Mistral.

## Quick Start

To run my project, install dependencies from `requirements.txt` and set up API keys in `.env` (see `SETUP.md` for detailed installation instructions).

**Streamlit Web UI:**
```bash
streamlit run app.py
```

**Command Line Interface:**
```bash
python -m src.finbrief AAPL --duke-gateway
```

See `SETUP.md` for comprehensive setup instructions, API key configuration, and testing guidelines.

## Video Links

- **Demo Video:** [Placeholder - Coming Soon]
- **Technical Walkthrough:** [Placeholder - Coming Soon]

## Evaluation

### RAG System Evaluation

**Qualitative Comparison: RAG vs. No-RAG**

My Streamlit application includes a side-by-side comparison feature that demonstrates the clear quality difference between RAG-augmented and non-RAG reports:

- **RAG Mode**: Provides specific, grounded information extracted from SEC filings with detailed citations
- **No-RAG Mode**: Generates responses without specific filing details

**RAG Pipeline Metrics**
- **Embedding Model**: all-MiniLM-L6-v2 (384-dimensional semantic embeddings)
- **Vector Search**: FAISS IndexFlatIP with L2-normalized cosine similarity
- **Retrieval**: Top-k semantic search over chunked SEC filing sections
- **Context Integration**: Dynamic chunk retrieval based on query relevance

### Sentiment Analysis Model

**DistilBERT Fine-Tuning Results** (Financial PhraseBank Dataset)

| Metric | Value |
|--------|-------|
| Training Accuracy | 95.2% |
| Validation Accuracy | 93.05% |
| Test Accuracy | 92.8% |
| Training Loss (Final) | 0.142 |
| Validation Loss (Final) | 0.198 |

**Model Configuration & Hyperparameter Tuning**
- Base Model: distilbert-base-uncased (67M parameters)
- Dataset: Financial PhraseBank (4,840 labeled sentences)
- Labels: Positive, Neutral, Negative
- Training Split: 80% train, 10% validation, 10% test
- Batch Size: 16
- Learning Rate: 2e-5
- Epochs: 3
- Optimizer: AdamW
- Training Environment: Google Colab with GPU acceleration

**Classification Performance by Class**
- Positive: High precision/recall on bullish financial statements
- Neutral: Balanced performance on factual statements
- Negative: Strong identification of bearish language

**Training Visualizations**

Confusion Matrix:

![DistilBERT Confusion Matrix](models/distillbert-fine-tuning/results/confusion_matrix.png)

ROC Curves:

![DistilBERT ROC Curves](models/distillbert-fine-tuning/results/roc_curves.png)

See `notebooks/sentiment_analysis_training.ipynb` for detailed training process.

### Model Architecture

| Component | Model | Parameters |
|-----------|-------|------------|
| Embedding | all-MiniLM-L6-v2 | 22M |
| Sentiment Classifier | DistilBERT (fine-tuned) | 67M |
| Generator (Gateway) | GPT-4.1 (Duke AI) | Frontier |
| Generator (Local Fallback) | TinyLlama-1.1B-Chat | 1.1B |

### Test Results

```
48 passed, 5 skipped (network tests)
```

## Individual Contributions

I (Arnav Meduri) am the sole contributor to this project.

## Project Structure

```
CS372/
├── README.md                 # Project overview
├── ATTRIBUTION.md            # AI and resource attribution
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── app.py                    # Streamlit web application
├── setup_venv.sh             # Setup script (Unix/Mac)
├── setup_venv.ps1            # Setup script (Windows)
│
├── src/                      # Source code
│   ├── finbrief.py           # Main application orchestration
│   ├── rag_system.py         # RAG pipeline (FAISS + embeddings)
│   ├── sentiment_classifier.py # DistilBERT sentiment analysis
│   │
│   ├── clients/              # API clients
│   │   ├── sec_edgar_client.py   # SEC EDGAR API client
│   │   ├── edgartools_client.py  # edgartools wrapper
│   │   ├── finnhub_client.py     # Finnhub metrics client
│   │   └── duke_gateway_model.py # Duke AI Gateway integration
│   │
│   └── utils/                # Utilities
│       ├── educational_brief.py  # Output data structures
│       ├── rich_formatter.py     # Report formatting
│       ├── prompt_loader.py      # Prompt management
│       ├── section_validator.py  # SEC section validation
│       ├── balance_sheet_analyzer.py # Financial ratio analysis
│       └── model_handler.py      # Local model handling
│
├── data/                     # Data and scripts
│   ├── test_finnhub.py       # Finnhub API test
│   ├── test_duke_gateway.py  # Duke Gateway test
│   └── test_any_company.py   # Company analysis test
│
├── models/                   # Trained models
│   └── distillbert-fine-tuning/  # Sentiment classifier
│       ├── model.safetensors
│       ├── config.json
│       ├── tokenizer files
│       └── results/          # Training visualizations
│           ├── confusion_matrix.png
│           └── roc_curves.png
│
├── notebooks/                # Jupyter notebooks
│   └── sentiment_analysis_training.ipynb
│
├── config/                   # Configuration
│   ├── prompts.md            # RAG mode prompts
│   └── prompts_no_rag.md     # No-RAG mode prompts
│
├── docs/                     # Documentation
│   └── figures/              # Training visualizations
│
├── tests/                    # Unit tests
│   ├── test_finnhub_client.py
│   ├── test_rag_system.py
│   └── test_sec_edgar_client.py
│
└── videos/                   # Demo and walkthrough videos
```

## Key Features

### RAG Pipeline
- **Embedding**: all-MiniLM-L6-v2 for semantic text embeddings
- **Vector Store**: FAISS for efficient similarity search
- **Data Sources**: SEC EDGAR filings (10-K, 10-Q) + Finnhub metrics
- **Retrieval**: Semantic search with L2-normalized cosine similarity

### Sentiment Analysis
- **Model**: Fine-tuned DistilBERT on Financial PhraseBank
- **Accuracy**: 93.05% validation accuracy
- **Integration**: Analyzes retrieved SEC filing chunks for sentiment distribution

### Report Generation
- **Primary**: Duke AI Gateway (GPT-4.1) for high-quality analysis
- **Fallback**: TinyLlama-1.1B-Chat for local inference
- **Output**: Structured educational briefs with transparent citations

### Section Validation & Processing
- **Validation**: Automatic quality checks for extracted SEC sections
- **Fallback Strategies**: Alternative extraction methods when primary fails
- **Balance Sheet Analysis**: 5 key financial ratios with interpretations

## License

This project is for educational and research purposes only.

See `ATTRIBUTION.md` for detailed attribution of AI-generated code, external libraries, models, and data sources.
