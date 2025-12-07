# Attribution

This document provides detailed attribution for AI-generated code, external libraries, datasets, APIs, and other resources used in FinBrief.

## AI-Generated Code

This project was developed with significant assistance from **Claude (Anthropic)**, an AI assistant. The following components were AI-generated with human oversight and modifications:

### Core Application Code

| File | AI Contribution | Human Modifications |
|------|-----------------|---------------------|
| `src/finbrief.py` | Full implementation, main orchestration | API key handling, error recovery |
| `src/sec_edgar_client.py` | Initial implementation, section extraction | User-Agent format, URL fixes |
| `src/finnhub_client.py` | Full implementation, rate limiting | Market cap formatting fixes |
| `src/definitions_corpus.py` | Full implementation, search logic | Term curation |
| `src/rag_system.py` | Vector embedding, FAISS indexing | Memory optimizations |
| `src/model_handler.py` | GPT-2 integration, LoRA support | Quantization, device selection |
| `src/educational_brief.py` | Structured output format, parsing | Section formatting |
| `src/confidence_head.py` | MLP architecture, calibration metrics | Heuristic estimator |
| `src/lora_trainer.py` | LoRA configuration, training loop | Hyperparameter tuning |
| `src/training_data_builder.py` | Training example generation | Example templates |
| `src/finetune.py` | CLI interface, workflow | - |
| `src/newsapi_client.py` | Full implementation | - |

### Tests

| File | AI Contribution |
|------|-----------------|
| `tests/test_definitions_corpus.py` | Full implementation (13 tests) |
| `tests/test_finnhub_client.py` | Full implementation (15 tests) |
| `tests/test_rag_system.py` | Full implementation (16 tests) |
| `tests/test_sec_edgar_client.py` | Full implementation (9 tests) |

### Configuration and Scripts

| File | AI Contribution |
|------|-----------------|
| `run.sh`, `run.py` | Full implementation |
| `setup_venv.sh`, `setup_venv.ps1` | Full implementation |
| `requirements.txt` | Dependency selection |
| `.env.example` | Template creation |
| `data/definitions/terms.json` | Term definitions (40 terms) |

### Documentation

| File | AI Contribution |
|------|-----------------|
| `README.md` | Full implementation |
| `SETUP.md` | Full implementation |
| `ATTRIBUTION.md` | Template and structure |
| `IMPLEMENTATION_PLAN.md` | Full implementation |
| `design.md` | Proposal structure (human content) |

## External Libraries

### Machine Learning & NLP

| Library | Version | License | Purpose |
|---------|---------|---------|---------|
| [PyTorch](https://pytorch.org/) | ≥2.1.0 | BSD-3-Clause | Deep learning framework |
| [Transformers](https://huggingface.co/transformers) | ≥4.35.0 | Apache 2.0 | LLM model loading and inference |
| [Sentence-Transformers](https://www.sbert.net/) | ≥2.2.0 | Apache 2.0 | Text embeddings for RAG |
| [FAISS](https://github.com/facebookresearch/faiss) | ≥1.7.4 | MIT | Vector similarity search |
| [PEFT](https://github.com/huggingface/peft) | ≥0.7.0 | Apache 2.0 | LoRA fine-tuning |
| [Datasets](https://huggingface.co/docs/datasets) | ≥2.14.0 | Apache 2.0 | Training data handling |
| [Accelerate](https://huggingface.co/docs/accelerate) | ≥0.24.0 | Apache 2.0 | Training acceleration |

### Data Processing

| Library | Version | License | Purpose |
|---------|---------|---------|---------|
| [Pandas](https://pandas.pydata.org/) | ≥2.1.0 | BSD-3-Clause | Data manipulation |
| [NumPy](https://numpy.org/) | ≥1.24.0 | BSD-3-Clause | Numerical computing |
| [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) | ≥4.12.0 | MIT | HTML parsing |
| [lxml](https://lxml.de/) | ≥4.9.0 | BSD-3-Clause | XML/HTML processing |
| [html2text](https://github.com/Alir3z4/html2text) | ≥2020.1.16 | GPL-3.0 | HTML to text conversion |

### API & Web

| Library | Version | License | Purpose |
|---------|---------|---------|---------|
| [Requests](https://requests.readthedocs.io/) | ≥2.31.0 | Apache 2.0 | HTTP requests |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | ≥1.0.0 | BSD-3-Clause | Environment variable loading |

### Testing

| Library | Version | License | Purpose |
|---------|---------|---------|---------|
| [pytest](https://pytest.org/) | ≥7.0.0 | MIT | Testing framework |

## Pre-trained Models

### Embedding Model

| Model | Source | License | Purpose |
|-------|--------|---------|---------|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Sentence-Transformers | Apache 2.0 | 384-dim text embeddings |

### Language Models

| Model | Source | License | Purpose |
|-------|--------|---------|---------|
| [GPT-2 Medium](https://huggingface.co/gpt2-medium) | OpenAI/HuggingFace | MIT | Primary generator (355M) |
| [DistilGPT2](https://huggingface.co/distilgpt2) | HuggingFace | Apache 2.0 | Lightweight alternative (82M) |
| [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | TinyLlama/HuggingFace | Apache 2.0 | Alternative generator |

## External APIs

### SEC EDGAR

- **Provider:** U.S. Securities and Exchange Commission
- **URL:** https://www.sec.gov/edgar/sec-api-documentation
- **License:** Public Domain (U.S. Government Work)
- **Usage:** Fetching 10-K, 10-Q, and 8-K filings
- **Rate Limits:** 10 requests/second recommended
- **Requirements:** User-Agent header with identifying information

### Finnhub

- **Provider:** Finnhub Stock API
- **URL:** https://finnhub.io/
- **License:** Proprietary (API Terms of Service)
- **Usage:** Real-time financial metrics (P/E, EPS, Market Cap, etc.)
- **Rate Limits:** 60 calls/minute (free tier)
- **Requirements:** API key (free registration)

### NewsAPI (Optional)

- **Provider:** NewsAPI.org
- **URL:** https://newsapi.org/
- **License:** Proprietary (API Terms of Service)
- **Usage:** Fetching news articles about companies
- **Rate Limits:** 100 requests/day (free tier)
- **Requirements:** API key (free registration)

## Data Sources for Definitions Corpus

The financial term definitions were curated from:

### Investor.gov (SEC)

- **URL:** https://www.investor.gov/introduction-investing/investing-basics/glossary
- **License:** Public Domain (U.S. Government Work)
- **Terms:** Primary source for SEC-related terms (10-K, 10-Q, 8-K, etc.)

### Investopedia

- **URL:** https://www.investopedia.com/financial-term-dictionary-4769738
- **License:** Content used for educational paraphrasing
- **Terms:** General financial concepts (P/E ratio, EPS, market cap, etc.)

## Code References

### SEC EDGAR API Usage
- SEC API Documentation: https://www.sec.gov/edgar/sec-api-documentation
- SEC Developer Resources: https://www.sec.gov/developer

### FAISS Implementation
- FAISS GitHub: https://github.com/facebookresearch/faiss
- FAISS Tutorial: https://github.com/facebookresearch/faiss/wiki

### LoRA Fine-Tuning
- PEFT Documentation: https://huggingface.co/docs/peft
- LoRA Paper: https://arxiv.org/abs/2106.09685

### Apple Silicon Optimization
- PyTorch MPS Backend: https://pytorch.org/docs/stable/notes/mps.html
- Apple Developer Documentation: https://developer.apple.com/metal/pytorch/

## License Summary

This project uses components under the following licenses:
- **Apache 2.0:** Transformers, Sentence-Transformers, PEFT, Requests
- **MIT:** FAISS, GPT-2, pytest
- **BSD-3-Clause:** PyTorch, Pandas, NumPy, python-dotenv
- **GPL-3.0:** html2text

All dependencies are compatible with educational and research use.

## Acknowledgments

- **Anthropic** for Claude AI assistance in development
- **HuggingFace** for hosting pre-trained models and PEFT library
- **U.S. Securities and Exchange Commission** for free access to SEC EDGAR filings
- **Finnhub** for financial metrics API
- **Meta AI Research** for FAISS vector search
- **Sentence-Transformers** team for embedding models
- **CS372 Course Staff** at Duke University
