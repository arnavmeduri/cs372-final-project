# Quick Reference: New CLI

## One-Line Comparison

```bash
# WITH RAG (default)
python -m src.finbrief IBM --duke-gateway -o ibm.txt

# WITHOUT RAG (just add one flag)
python -m src.finbrief IBM --no-rag --duke-gateway -o ibm_no_rag.txt
```

## All Options at a Glance

| What You Want | Command |
|---------------|---------|
| **Default (best quality)** | `python -m src.finbrief IBM --duke-gateway` |
| **Without RAG** | `python -m src.finbrief IBM --no-rag --duke-gateway` |
| **Without any external data** | `python -m src.finbrief IBM --no-rag --no-finnhub --duke-gateway` |
| **Standard mode (short)** | `python -m src.finbrief IBM --standard --duke-gateway` |
| **Save to file** | Add `-o filename.txt` to any command |
| **Markdown format** | Add `--format md` to any command |

## What Changed

| Before | After | Notes |
|--------|-------|-------|
| `--rich` required | Default now | Don't need flag anymore |
| Separate scripts | One script | Use `--no-rag` flag instead |
| Manual comparison | Easy comparison | Just toggle one flag |

## Data Sources Matrix

| Command Flags | SEC Filings | Finnhub | LLM Knowledge |
|---------------|-------------|---------|---------------|
| (default) | ✅ | ✅ | ✅ |
| `--no-rag` | ❌ | ✅ | ✅ |
| `--no-rag --no-finnhub` | ❌ | ❌ | ✅ |

## Quick Test

```bash
# See help
python -m src.finbrief --help

# Test default
python -m src.finbrief AAPL --duke-gateway

# Test without RAG
python -m src.finbrief AAPL --no-rag --duke-gateway
```

---
**✅ Implementation Complete**

