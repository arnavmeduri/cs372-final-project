# CS372 Final Project: FinBrief - Educational Financial Analysis
Arnav Meduri

## What It Does

My project is an educational financial analysis tool that generates beginner-friendly investment reports for publicly traded companies. The inspiration for this project came from my own experience trying to understand different financial metrics and risk disclosures for my own investment planning. With the volume of financial information available online, it can be difficult to identify what actually matters without getting overwhelmed by complex financial jargon and terminology. The RAG system I developed retrieves authoritative information from SEC filings, mandatory financial and business reports that public and private companies submit to the U.S. Securities and Exchange Commission through the EDGAR database, as well as up-to-date financial metrics from Finnhub, and then performs sentiment analysis on the retrieved filing chunks using a fine-tuned DistilBERT model. The RAG system makes use of semantic embeddings (all-MiniLM-L6-v2) and FAISS for efficient similarity search with L2-normalized cosine similarity. Reports are generated using Duke AI Gateway, which provides access to frontier models such as GPT-4.1 and Mistral.

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

- **Project Videos:** [https://drive.google.com/drive/folders/145TbuL4B_M3MqTBJTfBxpx0UDz7skKBy?usp=sharing](https://drive.google.com/drive/folders/145TbuL4B_M3MqTBJTfBxpx0UDz7skKBy?usp=sharing)

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

I fine-tuned a DistilBERT model on Google Colab specifically for financial text sentiment analysis. The model was trained on the Financial PhraseBank dataset (4,840 labeled financial sentences) to classify text as positive, neutral, or negative. The system analyzes each chunk of retrieved SEC filing text and provides an overall sentiment distribution (e.g., "Mixed: 34% Positive, 43% Neutral, 23% Negative") that helps investors understand the tone and outlook expressed in the company's official filings. This sentiment analysis is displayed as part of the generated reports.

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
- Model weights saved in `models/` directory

**Classification Performance by Class**
- Positive: High precision/recall on bullish financial statements
- Neutral: Balanced performance on factual statements
- Negative: Strong identification of bearish language

**Training Visualizations**

Confusion Matrix:

![DistilBERT Confusion Matrix](models/results/confusion_matrix.png)

ROC Curves:

![DistilBERT ROC Curves](models/results/roc_curves.png)

See `notebooks/sentiment_analysis_training.ipynb` for detailed training process and implementation.

### Model Architecture

| Component | Model | Parameters |
|-----------|-------|------------|
| Embedding | all-MiniLM-L6-v2 | 22M |
| Sentiment Classifier | DistilBERT (fine-tuned) | 67M |
| Generator | GPT-4.1 (Duke AI Gateway) | Frontier |

### Validation

I have tested the RAG pipeline with multiple companies across different sectors (technology, finance, consumer goods, healthcare) by comparing the specific insights from growth opportunities/risks in the generated reports with the original SEC filings to verify how much information is accurately captured. The sentiment analysis model performs well at capturing the tone/outlook expressed in SEC filings.

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
├── data/                     # Data files
│   └── Sentences_75Agree.txt # Financial PhraseBank dataset
│
├── models/                   # Trained models
│                                  # Sentiment classifier model files
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
│   ├── QUICK_CLI_REFERENCE.md
│   └── cleanup_cache.sh
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
- **Model**: Duke AI Gateway (GPT-4.1) for high-quality analysis
- **Output**: Structured educational briefs with transparent citations

### Section Validation & Processing
- **Validation**: Automatic quality checks for extracted SEC sections
- **Fallback Strategies**: Alternative extraction methods when primary fails
- **Balance Sheet Analysis**: 5 key financial ratios with interpretations

## License

This project is for educational and research purposes only.

See `ATTRIBUTION.md` for detailed attribution of AI-generated code, external libraries, models, and data sources.
