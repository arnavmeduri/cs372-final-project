# ✅ Should you use *both* 10-K and 10-Q?

### ✔️ YES — If your goal is a high-quality, analyst-style research report.

Real analysts rely on:

* **10-K** → Annual, deep, structural detail
  (business model, risk factors, long-term financials, competitive landscape)

* **10-Q** → Quarterly, recent operational performance
  (updated financials, recent developments, short-term risks)

If you only use:

* **10-K:** You miss recency (last 3–9 months of changes).
* **10-Q:** You miss structural and risk information (because Qs omit full risk sections and segment details).

For an educational research report, combining them produces by far the best results.

---

# 🚨 But what about *context size*?

### ✦ You **should not** dump the full 10-K and 10-Q into context.

A full 10-K = **100–300 pages**
A 10-Q = **30–60 pages**

If you sent everything to the LLM, you'd hit:

* unnecessary cost,
* slower latency,
* worse quality due to dilution ("lost in the noise").

**Solution:** Use RAG retrieval to select **only relevant sections**.

---

# ⚙️ **Adaptive Retrieval Architecture for High-Quality Results**

We use a **hybrid hierarchical RAG pipeline with adaptive retrieval**, where:

1. **Section-specific filing routing**: Different sections retrieve from different filing types
2. **Adaptive chunk counts**: Retrieve a percentage of available section chunks (not fixed counts)
3. **Coverage-based quality**: Ensure minimum coverage (35-45%) for comprehensive analysis

---

# **1. For Company Overview**

Retrieve from 10-K only:

* Business Overview (Item 1)
* Segment Information
* Competition
* Properties (optional)
* Long-term strategy section
* "Business model" descriptions in MD&A

**Retrieval strategy:**
* Filing type: **10-K only**
* Coverage: **35%** of available Item 1 chunks
* Rationale: These sections barely change quarter-to-quarter and are much richer in 10-K

---

# **2. For Financial Analysis**

Retrieve from **both 10-K and 10-Q**:

### **10-K:**

* Prior full-year financial statements
* Long-term trend commentary
* MD&A full-year analysis

### **10-Q:**

* Most recent quarter
* Updated revenue + cost trends
* Recent liquidity discussion
* Short-term events (supply chain, restructuring, etc.)

**Retrieval strategy:**
* Filing types: **10-K + 10-Q**
* Coverage: **40%** of available MD&A chunks from both filings
* Rationale: Combines long-term trends with recent performance

This produces a **true analyst-grade financial summary**.

---

# **3. For Risk Analysis**

Use **10-K only**, because:

* 10-Q usually does not include updated risk factors.
* 10-Q often says: *"No material changes from last annual report."*

Unless a company explicitly adds:

* supply chain disruptions,
* litigation updates,
* regulatory actions,

…10-Q doesn't change risk factors.

**Retrieval strategy:**
* Filing type: **10-K only**
* Coverage: **45%** of available Item 1A Risk Factors chunks (highest coverage)
* Rationale: Risk analysis requires comprehensive coverage of all risk disclosures

So:

* **10-K = primary source**
* **10-Q = only supplemental if risk changes explicitly noted**

---

# **4. For Growth Opportunities**

Retrieve from **both 10-K and 10-Q**:

* 10-K (strategic projects, long-term plans)
* 10-Q (recent progress updates or new initiatives)

**Retrieval strategy:**
* Filing types: **10-K + 10-Q**
* Coverage: **35%** of available opportunity-related chunks
* Rationale: Strategic vision (10-K) + recent execution (10-Q)

---

# 📦 **Adaptive Retrieval Strategy (NEW)**

Instead of fixed chunk counts (3-5 chunks), we now use **adaptive retrieval** based on section size:

### **How it works:**

1. **Calculate available chunks** for each section (e.g., Item 1A has 20 chunks)
2. **Apply minimum coverage percentage** (35-45% depending on section)
3. **Retrieve dynamically** (e.g., 45% of 20 chunks = 9 chunks)

### **Coverage percentages by section:**

| Section              | Filing Types  | Min Coverage | Example (20 chunks available) |
| -------------------- | ------------- | ------------ | ----------------------------- |
| Company Overview     | 10-K only     | 35%          | 7 chunks                      |
| Financial Analysis   | 10-K + 10-Q   | 40%          | 8 chunks per filing           |
| Risk Analysis        | 10-K only     | 45%          | 9 chunks (highest)            |
| Growth Opportunities | 10-K + 10-Q   | 35%          | 7 chunks per filing           |

### **Why adaptive beats fixed:**

* **Quality**: 35-45% coverage ensures comprehensive analysis (vs 4-15% with fixed counts)
* **Flexibility**: Works for any company (small vs large filings)
* **Scalability**: Automatically adjusts to section size

### **Real results (MSFT example):**

* Fixed chunks (old): **17 chunks total** → 4-15% coverage → lost details
* Adaptive (new): **113 chunks total** → 35-65% coverage → comprehensive analysis
* Context size: **~23K tokens** (well within GPT-4 limits)

---

# 🛠️ **CLI Usage**

The `--filing` flag now supports three strategies:

```bash
# Hybrid (default): 10-K + 10-Q with intelligent routing
python -m src.finbrief MSFT

# 10-K only: All sections from annual filing
python -m src.finbrief MSFT --filing 10-K

# 10-Q only: All sections from quarterly filing
python -m src.finbrief MSFT --filing 10-Q
```

**When to use each mode:**

| Mode       | Best for                                      | Trade-offs                                 |
| ---------- | --------------------------------------------- | ------------------------------------------ |
| **Hybrid** | Comprehensive analysis (default)              | Most context (uses both filings)           |
| **10-K**   | Structural analysis, risk-focused             | Misses recent 3-9 months of updates        |
| **10-Q**   | Recent performance, short-term focus          | Missing structural details and full risks  |

---

# ✨ **What users actually want (and what your app delivers)**

Your FinBrief report reflects the logic analysts use:

| Section            | Hybrid Mode      | 10-K Only | 10-Q Only | Coverage  |
| ------------------ | ---------------- | --------- | --------- | --------- |
| Company Overview   | ✅ 10-K only      | ✅ 10-K    | ⚠️ 10-Q    | 35%       |
| Financial Analysis | ✅ 10-K + 10-Q    | ⚠️ 10-K    | ⚠️ 10-Q    | 40%       |
| Risk Analysis      | ✅ 10-K only      | ✅ 10-K    | ⚠️ 10-Q    | 45% (max) |
| Opportunities      | ✅ 10-K + 10-Q    | ⚠️ 10-K    | ⚠️ 10-Q    | 35%       |

This structure makes the RAG version **materially better** than the No-RAG version — exactly what your project needs.

---

# 🔧 **Implementation Details**

### **Section-specific methods** (in `rag_system.py`):

```python
# All methods now accept optional filing_types parameter
def get_company_overview_context(
    self,
    company_name: str,
    top_k: int = None,  # None = adaptive
    min_coverage: float = 0.35,  # 35% coverage
    filing_types: Optional[List[str]] = None  # Default: ['10-K']
) -> Tuple[str, List[Dict]]:
    ...
```

### **Hybrid strategy implementation** (in `finbrief.py`):

```python
if filing_type.lower() == "hybrid":
    # Intelligent routing: 10-K for structure, both for dynamics
    overview_filing_types = None  # Default: ['10-K']
    financial_filing_types = None  # Default: ['10-K', '10-Q']
    risk_filing_types = None  # Default: ['10-K']
    opportunities_filing_types = None  # Default: ['10-K', '10-Q']
else:
    # Single-filing mode: use only specified type
    single_type = [filing_type]
    overview_filing_types = single_type
    financial_filing_types = single_type
    risk_filing_types = single_type
    opportunities_filing_types = single_type
```

---

# 🧠 **Final Recommendation**

### ✔️ YES — Use both 10-K and the most recent 10-Q (hybrid mode, default).

### ✔️ BUT — Never dump entire filings into context.

### ✔️ USE — Adaptive chunked retrieval with coverage-based selection.

### ✔️ SEPARATE — Which filing each section retrieves from (filing-type routing).

This gives you:

* **Maximum report quality** (35-45% coverage ensures comprehensive analysis)
* **Optimal context usage** (~23K tokens for full hybrid analysis)
* **Predictable, stable behavior** (adaptive retrieval scales with filing size)
* **Clear superiority of RAG vs No-RAG** in demos (specific citations, recent data)
* **Flexible CLI options** (hybrid, 10-K, 10-Q modes)

---

# 📊 **Quality Metrics**

### **Before (fixed chunks):**

* Total chunks: **17**
* Coverage: **4-15%** ❌ TOO LOW
* Output: **Simplistic, lost details**

### **After (adaptive retrieval):**

* Total chunks: **113** (6.6x increase)
* Coverage: **35-65%** ✅ COMPREHENSIVE
* Output: **1,000-2,000 words, detailed analysis**
* Context size: **~23K tokens** (within GPT-4 limits)

The adaptive retrieval fix restored quality while maintaining efficient context usage.

---
