# Attribution

## Dataset Attribution

**Financial PhraseBank** - Malo et al., "Good debt or bad debt: Detecting semantic orientations in economic texts"
- Source: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
- Paper: https://arxiv.org/abs/1307.5336
- File: `data/Sentences_75Agree.txt`
- Usage: Training data for DistilBERT sentiment classifier (4,840 labeled financial sentences)

## AI Generated Attributions

As part of this project, I primarily used AI for processing SEC filings, API integrations, and error handling. The overall architecture, RAG pipeline design, and component selection were my own design decisions.

**Note on SEC Processing:** Through my experience working on this project, I learned that SEC 10-K/10-Q processing is extremely complicated (inconsistent filing formats/document structures across companies and years and XBRL artifacts)! To address this, I implemented different handlers and validators (with the assistance of AI tools) to account for many different types of edge cases/fallback strategies -- AI was helpful in iterating through these different edge cases.

All AI generated code is attributed at the file level at the very top of each file. The majority of AI assistance was used for API integrations, SEC filing processing, and error handling.

The files in which the majority of the code was AI generated are as follows:

**Client Files (API Integrations)**
- `src/clients/sec_edgar_client.py`
- `src/clients/edgartools_client.py`
- `src/clients/finnhub_client.py`
- `src/clients/duke_gateway_model.py`

**Utils Files (Processing & Formatting)**
- `src/utils/section_validator.py`
- `src/utils/balance_sheet_analyzer.py`
- `src/utils/educational_brief.py`
- `src/utils/rich_formatter.py`
- `src/utils/prompt_loader.py`
- `src/utils/model_handler.py`

**Core Application**
- `src/finbrief.py` - API integration logic, error handling, fallback strategies, data processing workflows

## External Libraries

- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)
- [lxml](https://lxml.de/)
- [Requests](https://requests.readthedocs.io/)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
- [Streamlit](https://github.com/streamlit/streamlit)
- [edgartools](https://github.com/dgunning/edgartools)

## Pre-trained Models

- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Semantic embeddings for RAG retrieval ([Sentence-BERT paper](https://arxiv.org/abs/1908.10084))
- [GPT-4.1](https://platform.openai.com/docs/models) - Primary LLM for report generation (via Duke AI Gateway)
- [Mistral 7B](https://mistral.ai/) - Alternative LLM (via Duke AI Gateway)
- [DistilBERT](https://huggingface.co/distilbert-base-uncased) - Financial sentiment classification ([paper](https://arxiv.org/abs/1910.01108)), fine-tuned on Financial PhraseBank

## External APIs

- [SEC EDGAR API](https://www.sec.gov/edgar/sec-api-documentation) - Fetching 10-K, 10-Q company filings
- [Finnhub Stock API](https://finnhub.io/) - Real-time financial metrics (P/E ratio, EPS, Market Cap, etc.)
- [Duke AI Gateway](https://ai.cs.duke.edu/) - Access to frontier LLMs (GPT-4.1, Mistral on-site)

## Acknowledgments

- **U.S. Securities and Exchange Commission** for free public access to SEC EDGAR filings
- **Duke University** for LiteLLM Gateway access to frontier models
- **HuggingFace** for hosting pre-trained models
- **Finnhub** for financial metrics API
- **Meta AI Research** for FAISS
- **Malo et al.** for Financial PhraseBank dataset

