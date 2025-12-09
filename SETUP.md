# SETUP.md

## Overview

You will need to:
- Create a Python environment (Python 3.8+)
- Activate the environment
- Install dependencies from requirements.txt
- Set up API keys in .env file
- Run the Streamlit app or CLI

## Environment

Create environment with:
```bash
python3 -m venv venv
```

Activate environment:
- macOS/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

Install dependencies:
```bash
pip install -r requirements.txt
```

## API Keys

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:

**Required:**
- `LITELLM_TOKEN` - Duke AI Gateway token (for LLM generation)
  - Get your token at: https://dashboard.ai.duke.edu/
  - Log in with Duke credentials
  - Navigate to "AI Gateway" to access your API token
- `SEC_EDGAR_NAME` - Your name (required by SEC EDGAR API)
- `SEC_EDGAR_EMAIL` - Your email (required by SEC EDGAR API)

**Optional but Recommended:**
- `FINNHUB_API_KEY` - Finnhub API key (get free key at https://finnhub.io/)

Example `.env` file:
```
LITELLM_TOKEN=your_duke_ai_gateway_token_here
SEC_EDGAR_NAME=Your Name
SEC_EDGAR_EMAIL=your.email@example.com
FINNHUB_API_KEY=your_finnhub_api_key_here
```

## Running the Project

### Streamlit Web UI (Recommended)

**Note:** This application must be run locally and cannot be deployed to Streamlit Community Cloud due to file size limitations. The fine-tuned DistilBERT model weights (`model.safetensors`, ~250MB) are required for sentiment analysis inference and exceed Streamlit's deployment file size limits.

Run locally:
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

Features:
- Side-by-side RAG vs. No-RAG comparison
- Real-time company analysis
- Visual sentiment analysis results

For a demonstration of the web interface, see the demo video in the README.md.

### Command Line Interface

```bash
python -m src.finbrief AAPL --duke-gateway
```

Replace `AAPL` with any stock ticker.

## Recommended Things to Try Out

**Try different companies:**
- Large cap tech: `AAPL`, `MSFT`, `GOOGL`
- Finance: `JPM`, `BAC`, `GS`
- Healthcare: `JNJ`, `PFE`, `UNH`
- Consumer: `COST`, `WMT`, `NKE`

**Compare RAG vs. No-RAG:**
- In Streamlit, toggle "Compare with No-RAG mode"
- See how RAG provides specific, grounded information from SEC filings
- No-RAG mode generates more generic responses

**Explore sentiment analysis:**
- Reports include sentiment distribution of SEC filing chunks (Positive/Neutral/Negative)
- Based on DistilBERT model fine-tuned on Financial PhraseBank dataset
- Trained on Google Colab with GPU acceleration (93.05% validation accuracy)
- Analyzes the tone and outlook expressed in company's official filings
- See `notebooks/sentiment_analysis_training.ipynb` for training details

**View SEC filing citations:**
- All factual claims include source citations
- Links to original SEC filings (10-K, 10-Q)

## For Graders

### Testing Individual Components

Test RAG system:
```python
from src.rag_system import RAGSystem
rag = RAGSystem()
print("RAG system loaded successfully")
```

Test sentiment classifier:
```python
from src.sentiment_classifier import SentimentClassifier
classifier = SentimentClassifier()
result = classifier.classify_text("The company reported strong revenue growth.")
print(result)
```

Test SEC filing retrieval:
```python
from src.clients.edgartools_client import EdgarToolsClient
client = EdgarToolsClient(name="Test", email="test@example.com")
filings = client.get_annual_filings("AAPL", limit=1)
print(f"Retrieved {len(filings)} filing(s)")
```

## Troubleshooting

**ModuleNotFoundError:**
- Activate virtual environment: `source venv/bin/activate`
- Reinstall: `pip install -r requirements.txt`

**Duke AI Gateway errors:**
- Verify `LITELLM_TOKEN` in `.env`
- Check network access to Duke services

**Finnhub API errors:**
- Verify API key in `.env`
- Check rate limits (60 calls/minute free tier)
- System works without Finnhub (metrics unavailable)

**SEC EDGAR access issues:**
- Ensure `SEC_EDGAR_NAME` and `SEC_EDGAR_EMAIL` are set
- Valid contact info required by SEC API

**Verify installation:**
```bash
python -c "import torch; import transformers; import sentence_transformers; import faiss; import streamlit; print('All packages installed!')"
```

## Model Files

**DistilBERT Sentiment Classifier:**
- Fine-tuned model weights are stored in `models/model.safetensors`
- **Note:** This file is too large to push to GitHub (~250MB)
- **Download link:** [https://drive.google.com/file/d/13On6wG39REoUNH3rR5wAh7gvtIf-DhRi/view?usp=sharing](https://drive.google.com/file/d/13On6wG39REoUNH3rR5wAh7gvtIf-DhRi/view?usp=sharing)
- Place the downloaded file in the `models/` directory

**Other models automatically downloaded on first use:**
- **all-MiniLM-L6-v2** - Sentence embeddings (22M params)
- Models cached in `~/.cache/huggingface/`

## Data Files

- `data/Sentences_75Agree.txt` - Financial PhraseBank dataset (4,840 labeled sentences)
- Used for sentiment classifier training
- Included in repository

## Additional Resources

- `README.md` - Project overview and evaluation
- `ATTRIBUTION.md` - AI-generated code attribution
- `notebooks/sentiment_analysis_training.ipynb` - DistilBERT training process
- `config/prompts.md` - RAG mode LLM prompts
- `config/prompts_no_rag.md` - No-RAG mode LLM prompts
