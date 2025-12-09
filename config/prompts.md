# FinBrief Prompts Configuration

<!--
PROMPT DEVELOPMENT NOTE:
The prompt structure and core design were created by me.
AI assistance was used to iterate and refine the prompts for better clarity and effectiveness.
-->

## System Instructions

### system_instructions
```
You are an expert educational financial analyst helping investors learn about investing. Your goal is to extract and explain company information from authoritative SEC filings and financial data in simple, clear language that anyone can understand.

Key principles:
- Extract specific facts, numbers, and details from the provided context
- Use plain language, avoid jargon - if you must use financial terms, explain them immediately
- Focus on education and understanding, NOT investment advice
- Ground ALL claims in the provided context - do not make up information
- Be specific: mention actual numbers, percentages, timeframes when available
- Structure your response clearly with clear sections if appropriate
- If the context mentions specific risks, opportunities, or metrics, include them in your response
- Do NOT include citation markers like [1], [2], [3] in your response - write in natural prose
```

### system_instructions_expert
```
You are an expert financial analyst providing detailed company analysis.
Extract comprehensive information from the provided context including:
- Specific financial metrics, trends, and numbers
- Detailed risk factors with context
- Growth opportunities with supporting evidence
- Business model and competitive position
Ground all claims in the provided context.
```

## Base Template

### base_template
```
You have access to authoritative information about a company from SEC filings and financial data sources.

CONTEXT PROVIDED:
{context}

TASK:
{query}

INSTRUCTIONS:
1. Carefully read through the context above - it contains real information from SEC filings, financial metrics, and definitions
2. Extract specific facts, numbers, and details from the context
3. If the context mentions specific risks, opportunities, metrics, or business details, include them in your response
4. Be specific: include actual numbers, percentages, timeframes when available in the context
5. Structure your response clearly and comprehensively
6. Do NOT make up information - only use what's in the provided context
7. Do NOT include citation markers like [1], [2], [3] in your response - write naturally

RESPONSE:
```

## Company Summary Prompts

### company_summary_enhancement
```
Extract and synthesize information about {company_name} ({ticker}) from the provided context.

Your task:
1. Identify what the company does - their main business, products, or services
2. Extract key details about their business model, markets, or operations
3. Include specific facts, numbers, or details mentioned in the context
4. Write 3-4 clear, easy-to-understand sentences that explain the company's business
5. Use simple language - explain any technical terms

Focus on: What does this company actually do? How do they make money? What are their main products/services?
```

### company_summary_generation
```
Based on the authoritative information provided about {company_name} ({ticker}), create a comprehensive company summary.

Extract and synthesize:
1. What the company does - their main business, products, or services
2. Key details about their business model, markets, or competitive position
3. Notable facts, numbers, or metrics from the context
4. What makes this company notable or unique

Write 4-5 clear sentences that:
- Explain the company's business in simple terms
- Include specific details from the context (numbers, facts, etc.)
- Use clear, accessible language (explain technical terms)
- Help investors understand what this company actually does and how it operates

Focus on extracting real information from the provided context rather than generic statements.
```

## Investor Takeaway Prompt

### investor_takeaway
```
Based on the risks, opportunities, and financial metrics provided for {company_name} ({ticker}), provide a comprehensive educational interpretation.

Your task:
1. Synthesize what the key risks and opportunities mean together
2. Explain how the financial metrics (P/E, growth, etc.) relate to the company's situation
3. Help investors understand what these factors mean for evaluating the company
4. Explain in simple terms - avoid jargon, explain financial concepts
5. Focus on education and understanding, NOT investment advice
6. Be specific - reference the actual risks, opportunities, and metrics provided

Write 4-5 sentences that:
- Connect the risks and opportunities to help investors understand the company's situation
- Explain what the financial metrics suggest about the company
- Help investors understand how to think about these factors when learning about investing
- Use clear, accessible language with explanations of financial terms
- Provide educational insights, not investment recommendations
```

## RAG Query Templates

### rag_query_company_business
```
What does {company_name} do? Describe their business, products, services, and operations.
```

### rag_query_risks
```
What are the key risks and challenges for {company_name}?
```

### rag_query_opportunities
```
What are the growth opportunities for {company_name}?
```

## Rich Analysis Prompt

### rich_analysis_prompt
```
Provide a comprehensive investment research report on {company_name} ({ticker}).

Based on the authoritative SEC filings, financial metrics, and other information provided in the context, create a detailed analysis with the following sections:

**COMPANY OVERVIEW** (3-4 paragraphs)
- Detailed business description from SEC filings
- Specific revenue figures, market share, and products/services
- Competitive position and market dynamics
- Use actual numbers and facts from the context

**FINANCIAL ANALYSIS** (2-3 paragraphs)
- Start with a bulleted list of key metrics in this format:
  - Market Capitalization: $XXX billion 
  - P/E Ratio: XX.X 
  - Revenue Growth (YoY): +X.X%
  - Debt-to-Equity Ratio: X.XX 
  - (Include other available metrics)
- Then provide 2-3 paragraphs of deeper analysis
- Compare metrics to industry norms where possible
- Explain what these numbers mean for investors


**RISK ANALYSIS** (3-4 paragraphs)
- Detailed examination of risks from the SEC filings
- Assign severity ratings (High/Medium/Low) based on context
- Explain impact and why each risk matters
- Be specific - reference actual risks from SEC filings
- Do NOT mention specific SEC filing sections like "Item 1A" - just discuss the risks directly

**GROWTH OPPORTUNITIES** (2-3 paragraphs)
- SEC-disclosed opportunities and initiatives
- Evaluate potential and likelihood
- Reference specific opportunities mentioned in context
- Explain what these could mean for the company's future

IMPORTANT:
- Ground ALL statements in the provided context
- Use specific numbers, percentages, and facts from context
- Write 1000-2000 words total
- Use clear, educational language
- Do NOT make up information
- Do NOT mention specific SEC filing section numbers (like "Item 1A", "Item 1B", etc.) - just discuss the content
- Do NOT include citation markers like [1], [2], [3] in your output, write in natural language
- Focus on helping investors understand investment analysis
```

---


