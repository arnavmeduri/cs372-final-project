# Self-Assessment

## Category 1: Machine Learning 

**Completed project individually without a partner (10 pts)**
- Solo project completed by me (Arnav Meduri). 

**Used sentence embeddings for semantic similarity or retrieval (5 pts)**
- Used all-MiniLM-L6-v2 sentence embeddings for semantic retrieval in RAG system. See `src/rag_system.py` lines 39-42 (model initialization) and lines 233-262 (`build_index()` method using FAISS similarity search).

**Made API calls to state-of-the-art model (GPT-4, Claude, Gemini) with meaningful integration into your system (5 pts)**
- Integrated GPT-4.1 via Duke AI Gateway for educational report generation. See `src/clients/duke_gateway_model.py` lines 40-120 (`analyze_with_context()` method) and `src/finbrief.py` lines 178-223 (Duke Gateway model loading and integration).

**Built retrieval-augmented generation (RAG) system with document retrieval and generation components (10 pts)**
- Built RAG system retrieving from SEC EDGAR filings (10-K, 10-Q) with GPT-4.1 generation. Document retrieval in `src/rag_system.py` (FAISS indexing and semantic search), SEC filing integration in `src/clients/edgartools_client.py`, and generation in `src/clients/duke_gateway_model.py`. Full pipeline in `src/finbrief.py` (lines 1241-1550).

**Applied prompt engineering with evaluation of multiple prompt designs (3 pts)**
- Engineered separate prompts for RAG mode (`config/prompts.md` lines 11-23 for system instructions, lines 137-183 for rich analysis prompt) and No-RAG mode (`config/prompts_no_rag.md` lines 13-19 for system instructions, lines 23-100 for rich analysis prompt). See `docs/examples.md` for side-by-side qualitative comparison of example reports. 

**Collected or constructed original dataset through substantial engineering effort with documented methodology (10 pts)**
- Constructed dataset from SEC EDGAR filings API and Finnhub financial metrics API with substantial preprocessing for RAG pipeline. SEC filing retrieval and section extraction in `src/clients/edgartools_client.py` (lines 45-150 for filing retrieval, section parsing) and `src/clients/sec_edgar_client.py`. Section validation and quality filtering in `src/utils/section_validator.py` (lines 25-200). Financial metrics collection and formatting in `src/clients/finnhub_client.py` (lines 80-180). Data chunking and indexing for FAISS in `src/rag_system.py` (lines 50-90 for chunking, lines 233-262 for index building). Complete pipeline orchestration in `src/finbrief.py` (lines 1104-1213).

**Fine-tuned pretrained model on your dataset (5 pts)**
- Fine-tuned DistilBERT on Financial PhraseBank for financial sentiment classification. Training implementation in `notebooks/sentiment_analysis_training.ipynb` cells 8-12 (training loop), with final validation accuracy of 93.05%. Model weights saved in `models/`.

**Built multi-stage ML pipeline connecting outputs of one model to inputs of another (7 pts)**
- Built multi-stage pipeline: (1) MiniLM embeddings convert text to vectors (`src/rag_system.py` lines 39-42, 233-262), (2) FAISS similarity search retrieves relevant chunks (`src/rag_system.py` lines 268-296), (3) DistilBERT analyzes sentiment of retrieved chunks (`src/sentiment_classifier.py` lines 36-80), (4) GPT-4.1 generates reports using retrieved context and sentiment analysis (`src/clients/duke_gateway_model.py` lines 40-120). Complete pipeline in `src/finbrief.py` lines 1171-1360 showing data flow: SEC filings → embeddings → FAISS retrieval → sentiment analysis → LLM generation.

**Modular code design with reusable functions and classes rather than monolithic scripts (3 pts)**
- Modular architecture with separate classes: `RAGSystem` (`src/rag_system.py`), `SentimentClassifier` (`src/sentiment_classifier.py`), `DukeGatewayModel` (`src/clients/duke_gateway_model.py`), `EdgarToolsClient` (`src/clients/edgartools_client.py`), `FinnhubClient` (`src/clients/finnhub_client.py`).

**Implemented proper train/validation/test split with documented split ratios (3 pts)**
- 80% train, 10% validation, 10% test split implemented in `notebooks/sentiment_analysis_training.ipynb` cell 3. Split ratios documented in README.md Evaluation section.

**Tracked and visualized training curves showing loss and/or metrics over time (3 pts)**
- Training and validation loss/accuracy tracked over 3 epochs in `notebooks/sentiment_analysis_training.ipynb`. Visualizations include confusion matrix (`models/results/confusion_matrix.png`) and ROC curves (`models/results/roc_curves.png`) shown in README.md.

**Conducted systematic hyperparameter tuning using validation data (5 pts)**
- Hyperparameter tuning for DistilBERT: tested learning rates (2e-5, 3e-5, 5e-5), batch sizes (8, 16, 32), and epochs (3, 4, 5) using validation data. Final configuration: learning rate 2e-5, batch size 16, 3 epochs. See `notebooks/sentiment_analysis_training.ipynb` and README.md "Model Configuration & Hyperparameter Tuning" section.

**Implemented production-grade deployment (evidence of at least two considerations such as rate limiting, caching, monitoring, error handling, logging) (10 pts)**
- Implemented comprehensive production-grade features: (1) Rate limiting for Finnhub API (`src/clients/finnhub_client.py` lines 176-183, 186-201 with automatic retry on 429 errors), SEC EDGAR API (`src/clients/sec_edgar_client.py` line 252), and Duke Gateway (error handling lines 272-275); (2) Memory management with `clear_memory()` functions (`src/rag_system.py` lines 16-17, `src/utils/model_handler.py` lines 27-37) and KV caching (line 212); (3) Comprehensive error handling with 81+ try/except blocks across all modules; (4) Verbose logging system (`src/finbrief.py` lines 45, 50, 68-114, 117-120) with detailed progress tracking and monitoring; (5) Fallback strategies for section extraction (`src/utils/section_validator.py`), prompt loading (`src/utils/prompt_loader.py` lines 61-75), and RAG chunking (`src/rag_system.py` lines 189-190).

**Trained model using GPU/TPU/CUDA acceleration (5 pts)**
- DistilBERT fine-tuning performed on Google Colab with GPU acceleration (T4). Training environment documented in README.md and `notebooks/sentiment_analysis_training.ipynb`.

**Deployed model as functional web application with user interface (10 pts)**
- Implemented a Streamlit web application (`app.py`) with interactive UI for company analysis, side-by-side RAG vs. No-RAG comparison, and sentiment analysis visualization. Application runs locally (see SETUP.md for instructions). Note: Cannot deploy to Streamlit Community Cloud due to model file size constraints (~250MB), but demo video shows web interface functionality.

---

## Category 2: Following Directions (Maximum 20 points)

**Project submitted on 12/9 before the 5pm deadline**
- Project submitted before the 5pm deadline on 12/9.

**Self-assessment submitted that follows guidelines for at most 15 selections in Machine Learning with evidence**
- Self-assessment (this doc) follows guidelines and provides evidence for rubric items in the ML category.

**README.md file exists with required sections**
- README.md includes "What It Does" section, "Quick Start" section, "Video Links" section with project videos, "Evaluation" section with quantitative and qualitative results, and "Individual Contributions" section.

**SETUP.md exists with clear, step-by-step installation instructions**
- SETUP.md provides comprehensive setup instructions (environment creation, dependency installation, API key configuration, and component testing instructions for graders).

**ATTRIBUTION.md exists documenting AI-generated code and external resources**
- ATTRIBUTION.md documents all AI-generated code (attributed at file level), dataset sources, external libraries, pre-trained models, and external APIs used in the project.

**Demo video and technical walkthrough video**
- Video links section exists in README.md with Google Drive link to project videos.

**Project workshop attendance**
- Attended all 6 project workshop days.

**Repository organization follows requirements**
- Repository includes required directories: `src/` (source code), `data/` (datasets), `models/` (trained models), `notebooks/` (Jupyter notebooks), and `docs/` (documentation). Demo videos hosted on Google Drive (linked in README.md).

**Dependency management file included**
- `requirements.txt` file included, with all necessary dependencies listed.

## Category 3: Project Cohesion and Motivation (Maximum 20 points)

### Project Purpose and Motivation (3 points each)

**README clearly articulates a single, unified project goal or research question**
- README.md clearly states the unified goal: helping beginner investors understand publicly traded companies through RAG-augmented analysis of SEC filings and financial data.

**Project demo video effectively communicates why the project matters to a non-technical audience in non-technical terms**
- See demo video.

**Project addresses a real-world problem or explores a meaningful research question**
- The project addresses the real-world problem of beginner investors being overwhelmed by complex financial jargon and SEC filings when trying to understand publicly traded companies.

### Technical Coherence (3 points each)

**Technical walkthrough demonstrates how components work together synergistically (not just isolated experiments)**
- See technical walkthrough video.

**Project shows clear progression from problem → approach → solution → evaluation**
- See technical walthrough and demo videos.

**Design choices are explicitly justified in videos or documentation**
- See technical walkthrough video and SETUP.md for design choice justifications.

**Evaluation metrics directly measure the stated project objectives**
- README.md Evaluation section includes qualitative RAG vs. No-RAG comparison (measuring grounded information quality) and DistilBERT results (measuring sentiment analysis accuracy).

**None of the major components awarded rubric item credit in the machine learning category are superfluous to the larger goals of the project**
- All ML components (RAG retrieval, DistilBERT sentiment analysis, LLM generation) work together toward the unified goal of generating beginner-friendly educational reports. See codebase, demo video, and self-assessment ML category for evidence.

**Clean codebase with readable code and no extraneous, stale, or unused files**
- Codebase follows recommended file structure with no extraneous/stale/unused files. See codebase.

