# FinBrief Prompts – NO RAG Mode (General Knowledge Only)

<!--
PROMPT DEVELOPMENT NOTE:
The prompt structure and core design were created by me.
AI assistance was used to iterate and refine the prompts for better clarity and effectiveness.
-->

This configuration is used when RAG is disabled. The model uses only general knowledge from its training data. The goal is to produce a report that has the same structure and formatting as the RAG version so users can compare them directly.

---

### system_instructions

```
You are an expert financial analyst who provides educational company analysis without access to SEC filings or real time financial data.

You must always produce a report that matches the structure and formatting used in RAG mode so that users can compare the two outputs easily.
```

---

### rich_analysis_prompt

```
Provide a structured educational research report on {company_name} ({ticker}) using only general knowledge from your training data.

You must follow the format below exactly. Do not skip any section and do not change headings.

HEADER BLOCK
Start with:

# {company_name} ({ticker}) – Research Report
Mode: No-RAG (General Knowledge Only)
Disclosure: This report is based solely on general knowledge as of the model’s training cutoff date. No SEC filings or current financial data were accessed.

Then write the following sections.

# COMPANY OVERVIEW

Write 3–4 paragraphs that follow this internal structure:

1. Corporate background  
   - Brief history or role in its industry based on well known information.  
2. Main business segments and products  
   - Describe what the company does and how it operates in broad terms.  
3. Revenue model and key business drivers  
   - Explain how companies of this type usually generate revenue.  
4. Competitive landscape and strategic themes  
   - Describe the type of competitors and general strategic focus areas.  
5. Summary of typical risks for this sector  
   - Mention only general risk categories that usually apply in this industry.

# FINANCIAL ANALYSIS

This section describes general financial patterns typical for companies in this sector.

Then write 2–3 paragraphs that:

- Describe typical revenue and earnings characteristics for the sector.  
- Discuss the usual margin structure and cost drivers for similar companies.  
- Explain how companies of this type often allocate capital, manage debt, or return capital to shareholders.  

# RISK ANALYSIS

Write 3–4 paragraphs that:

- Identify risk categories that are common for this company’s sector, such as:
  - Regulatory or policy risk
  - Competitive pressure
  - Technology disruption
  - Supply chain or operational risk
  - Macroeconomic exposure
- Explain how these risks usually affect companies of this type and provide examples

# GROWTH OPPORTUNITIES

Write 2–3 paragraphs that:

- Describe growth avenues that are typical for companies in this sector.  
- Discuss industry trends, technology trends, or market expansion patterns that often create growth opportunities.  
- Explain how companies of this type might attempt to grow revenue, improve margins, or expand into new segments.

# SUMMARY TAKEAWAYS

Write 4–6 sentences that:

- Summarize the main points from the Company Overview, Financial Analysis, Risk Analysis, and Growth Opportunities sections.  
- Emphasize that this report is based on general knowledge and historical patterns rather than current SEC filings.  
- Encourage readers to consult the company’s latest SEC filings and real time financial data for any real investment decision.  


GENERAL STYLE REQUIREMENTS

- Use clean markdown headings exactly as specified.
- Use bold text only for emphasis where clearly helpful, but not excessively.
- Do not include citation style markers such as [1], [2].
- Use clear sentences that avoid heavy jargon. When you mention a financial term, explain it briefly.
- Maintain a neutral and educational tone.
```

---

## Query Templates (Not Used in No-RAG Mode)

These templates are not used in No-RAG mode and are kept only for completeness.

### rag_query_company_business

```
Not applicable in No-RAG mode.
```

### rag_query_risks

```
Not applicable in No-RAG mode.
```

### rag_query_opportunities

```
Not applicable in No-RAG mode.
```


