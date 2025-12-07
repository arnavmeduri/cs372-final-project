"""
Training data builder for FinBrief LoRA fine-tuning.

Generates training examples from:
- SEC filings (10-K, 10-Q)
- Definitions corpus
- Finnhub metrics (if available)

Target: 150-200 high-quality examples covering all output sections.
"""
import os
import json
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .sec_edgar_client import SECEdgarClient
from .definitions_corpus import DefinitionsCorpus
from .finnhub_client import FinnhubClient
from .educational_brief import (
    DifficultyLevel, RiskSeverity
)


@dataclass
class TrainingExample:
    """A single training example for LoRA fine-tuning."""
    input: str
    output: str
    section_type: str  # 'summary', 'risks', 'opportunities', 'metrics', 'terms', etc.
    ticker: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'input': self.input,
            'output': self.output,
            'section_type': self.section_type,
            'ticker': self.ticker
        }


class TrainingDataBuilder:
    """
    Builds training data for FinBrief LoRA fine-tuning.
    """
    
    # Template for different sections
    SECTION_TEMPLATES = {
        'summary': {
            'query': "Provide a beginner-friendly company summary for {company}.",
            'instruction': "Write a 2-3 sentence summary explaining what the company does in simple terms."
        },
        'risks': {
            'query': "What are the key risks students should understand about {company}?",
            'instruction': "List 2-3 risks with severity levels and beginner-friendly explanations."
        },
        'opportunities': {
            'query': "What are the growth opportunities for {company}?",
            'instruction': "List 2-3 opportunities with categories and brief explanations."
        },
        'metrics': {
            'query': "Explain these financial metrics for a beginner investor.",
            'instruction': "Format metrics and provide student-friendly interpretations."
        },
        'terms': {
            'query': "Explain these financial terms for someone new to investing.",
            'instruction': "Define each term in simple language with examples."
        },
        'takeaway': {
            'query': "What does this mean for a new investor considering {company}?",
            'instruction': "Provide educational insights without giving investment advice."
        }
    }
    
    # Company knowledge base for generating examples
    COMPANY_INFO = {
        'AAPL': {
            'name': 'Apple Inc.',
            'description': 'designs and sells consumer electronics including the iPhone, Mac, iPad, and Apple Watch',
            'services': 'App Store, iCloud, Apple Music, Apple TV+',
            'risks': ['competition', 'supply chain', 'regulatory'],
            'opportunities': ['services growth', 'wearables', 'emerging markets']
        },
        'MSFT': {
            'name': 'Microsoft Corporation',
            'description': 'develops software, cloud services, and hardware including Windows, Office 365, Azure, and Xbox',
            'services': 'Azure cloud, Microsoft 365, LinkedIn, GitHub',
            'risks': ['cloud competition', 'cybersecurity', 'antitrust'],
            'opportunities': ['AI integration', 'cloud growth', 'gaming']
        },
        'GOOGL': {
            'name': 'Alphabet Inc.',
            'description': 'operates the Google search engine, YouTube, Android, and Google Cloud',
            'services': 'Google Search, YouTube, Google Cloud, Android',
            'risks': ['advertising dependence', 'regulatory', 'competition'],
            'opportunities': ['AI/ML', 'cloud expansion', 'autonomous vehicles']
        },
        'AMZN': {
            'name': 'Amazon.com Inc.',
            'description': 'operates an e-commerce marketplace, AWS cloud services, and streaming entertainment',
            'services': 'Prime, AWS, Alexa, Prime Video',
            'risks': ['competition', 'labor costs', 'regulatory'],
            'opportunities': ['AWS growth', 'advertising', 'healthcare']
        },
        'NVDA': {
            'name': 'NVIDIA Corporation',
            'description': 'designs GPUs for gaming, data centers, AI, and automotive applications',
            'services': 'GeForce gaming, data center GPUs, CUDA platform',
            'risks': ['cyclical demand', 'competition', 'supply constraints'],
            'opportunities': ['AI boom', 'data center growth', 'automotive']
        },
        'TSLA': {
            'name': 'Tesla Inc.',
            'description': 'designs and manufactures electric vehicles, energy storage, and solar products',
            'services': 'Supercharger network, Full Self-Driving, Energy products',
            'risks': ['competition', 'production challenges', 'regulatory'],
            'opportunities': ['EV adoption', 'energy storage', 'autonomous driving']
        },
        'META': {
            'name': 'Meta Platforms Inc.',
            'description': 'operates social media platforms including Facebook, Instagram, and WhatsApp',
            'services': 'Facebook, Instagram, WhatsApp, Oculus VR',
            'risks': ['privacy regulations', 'competition', 'metaverse investments'],
            'opportunities': ['Reels growth', 'metaverse', 'AI']
        },
        'JPM': {
            'name': 'JPMorgan Chase & Co.',
            'description': 'provides investment banking, asset management, and consumer banking services',
            'services': 'Investment banking, Chase retail banking, Asset management',
            'risks': ['credit risk', 'interest rate sensitivity', 'regulatory'],
            'opportunities': ['wealth management', 'digital banking', 'international expansion']
        }
    }
    
    def __init__(
        self,
        sec_client: Optional[SECEdgarClient] = None,
        definitions: Optional[DefinitionsCorpus] = None,
        finnhub_client: Optional[FinnhubClient] = None
    ):
        """
        Initialize the training data builder.
        
        Args:
            sec_client: SEC EDGAR client (optional)
            definitions: Definitions corpus (optional)
            finnhub_client: Finnhub client (optional)
        """
        self.sec_client = sec_client
        self.definitions = definitions or DefinitionsCorpus()
        self.finnhub_client = finnhub_client
        
    def generate_summary_examples(self, num_examples: int = 20) -> List[TrainingExample]:
        """Generate company summary training examples."""
        examples = []
        
        for ticker, info in self.COMPANY_INFO.items():
            if len(examples) >= num_examples:
                break
                
            # Create input context
            input_text = f"""Context: {info['name']} {info['description']}. 
The company's key services include {info['services']}.

Query: {self.SECTION_TEMPLATES['summary']['query'].format(company=info['name'])}"""
            
            # Create output
            output_text = f"""1. Company Summary (Beginner-Friendly)

{info['name']} {info['description']}. The company also offers services like {info['services']}, 
which provide recurring revenue and help keep customers in the Apple ecosystem.

Citations: (10-K Item 1: Business Description)"""
            
            examples.append(TrainingExample(
                input=input_text,
                output=output_text,
                section_type='summary',
                ticker=ticker
            ))
        
        return examples
    
    def generate_risk_examples(self, num_examples: int = 30) -> List[TrainingExample]:
        """Generate risk analysis training examples."""
        examples = []
        
        risk_templates = {
            'competition': {
                'desc': 'faces intense competition in all markets it operates',
                'severity': 'Medium',
                'explanation': 'Many companies compete for the same customers. If competitors offer better products or lower prices, this company could lose market share.'
            },
            'supply chain': {
                'desc': 'relies on global supply chains that could be disrupted',
                'severity': 'Medium-High',
                'explanation': 'Products are made in factories around the world. Events like natural disasters, pandemics, or trade disputes could interrupt production.'
            },
            'regulatory': {
                'desc': 'subject to government regulations that could change',
                'severity': 'High',
                'explanation': 'Governments might pass new laws affecting how the company operates, what it can charge, or where it can do business.'
            },
            'cybersecurity': {
                'desc': 'vulnerable to data breaches and cyber attacks',
                'severity': 'High',
                'explanation': 'Hackers could steal customer data or disrupt operations. This could damage the company\'s reputation and lead to lawsuits.'
            },
            'cyclical demand': {
                'desc': 'experiences fluctuating demand based on economic conditions',
                'severity': 'Medium',
                'explanation': 'Sales go up and down with the economy. During recessions, customers buy less, which hurts revenue.'
            }
        }
        
        for ticker, info in self.COMPANY_INFO.items():
            for risk_type in info['risks'][:2]:
                if risk_type in risk_templates:
                    risk = risk_templates[risk_type]
                    
                    input_text = f"""Context: Risk Factors from {info['name']} SEC filing:
The company {risk['desc']}. This could materially affect business operations.

Query: {self.SECTION_TEMPLATES['risks']['query'].format(company=info['name'])}"""
                    
                    output_text = f"""4. Risks Students Should Understand

🟡 **{risk_type.title()} Risk** ({risk['severity']}): {info['name']} {risk['desc']}.

Student Explanation: {risk['explanation']}

Citations: (10-K Item 1A: Risk Factors)"""
                    
                    examples.append(TrainingExample(
                        input=input_text,
                        output=output_text,
                        section_type='risks',
                        ticker=ticker
                    ))
            
            if len(examples) >= num_examples:
                break
        
        return examples
    
    def generate_opportunity_examples(self, num_examples: int = 25) -> List[TrainingExample]:
        """Generate growth opportunity training examples."""
        examples = []
        
        opportunity_templates = {
            'services growth': {
                'category': 'Revenue Diversification',
                'desc': 'expanding high-margin services business',
                'explanation': 'Services like subscriptions generate recurring revenue with higher profit margins than hardware.'
            },
            'cloud growth': {
                'category': 'Technology',
                'desc': 'growing cloud computing market share',
                'explanation': 'More businesses are moving to the cloud, creating a large and growing market.'
            },
            'AI integration': {
                'category': 'Product Innovation',
                'desc': 'integrating AI capabilities across products',
                'explanation': 'AI features can make products more valuable and attract new customers.'
            },
            'emerging markets': {
                'category': 'Market Expansion',
                'desc': 'expanding presence in developing countries',
                'explanation': 'Billions of potential new customers in countries with growing middle classes.'
            },
            'AI boom': {
                'category': 'Industry Tailwind',
                'desc': 'benefiting from explosive growth in AI demand',
                'explanation': 'The rapid adoption of AI requires significant computing hardware, driving demand.'
            }
        }
        
        for ticker, info in self.COMPANY_INFO.items():
            for opp_type in info['opportunities'][:2]:
                if opp_type in opportunity_templates:
                    opp = opportunity_templates[opp_type]
                    
                    input_text = f"""Context: From {info['name']} SEC filing MD&A section:
Management believes the company is well-positioned for growth through {opp['desc']}.

Query: {self.SECTION_TEMPLATES['opportunities']['query'].format(company=info['name'])}"""
                    
                    output_text = f"""3. Opportunities (Based on SEC filings)

📈 **{opp['category']}**: {info['name']} is {opp['desc']}.

Why This Matters: {opp['explanation']}

Citations: (10-K Item 7: MD&A)"""
                    
                    examples.append(TrainingExample(
                        input=input_text,
                        output=output_text,
                        section_type='opportunities',
                        ticker=ticker
                    ))
            
            if len(examples) >= num_examples:
                break
        
        return examples
    
    def generate_metrics_examples(self, num_examples: int = 20) -> List[TrainingExample]:
        """Generate financial metrics interpretation examples."""
        examples = []
        
        # Sample metrics scenarios
        scenarios = [
            {
                'metrics': {'pe': 35, 'growth': 15, 'debt': 0.5},
                'interpretation': 'high growth expectations with low debt - typical growth stock profile'
            },
            {
                'metrics': {'pe': 12, 'growth': 3, 'debt': 1.8},
                'interpretation': 'lower valuation with modest growth and higher debt - value stock characteristics'
            },
            {
                'metrics': {'pe': 45, 'growth': 40, 'debt': 0.2},
                'interpretation': 'premium valuation justified by exceptional growth and minimal debt'
            },
            {
                'metrics': {'pe': 18, 'growth': 8, 'debt': 1.0},
                'interpretation': 'moderate valuation with steady growth and balanced debt levels'
            }
        ]
        
        for ticker, info in list(self.COMPANY_INFO.items())[:5]:
            for scenario in scenarios[:2]:
                m = scenario['metrics']
                
                input_text = f"""Context: Financial metrics for {info['name']}:
P/E Ratio: {m['pe']}, Revenue Growth: {m['growth']}%, Debt-to-Equity: {m['debt']}

Query: {self.SECTION_TEMPLATES['metrics']['query']}"""
                
                output_text = f"""2. Key Financial Metrics for Students

• P/E Ratio: {m['pe']} (Investors pay ${m['pe']} for every $1 of earnings)
• Revenue Growth: +{m['growth']}% (Sales grew {m['growth']}% vs last year)
• Debt-to-Equity: {m['debt']} ({'Low' if m['debt'] < 0.5 else 'Moderate' if m['debt'] < 1.5 else 'Higher'} debt level)

Student Interpretation:
These metrics suggest {scenario['interpretation']}. {'A high P/E means investors expect strong future growth.' if m['pe'] > 25 else 'A lower P/E might indicate the market has modest growth expectations.'}

Source: Finnhub Metrics"""
                
                examples.append(TrainingExample(
                    input=input_text,
                    output=output_text,
                    section_type='metrics',
                    ticker=ticker
                ))
            
            if len(examples) >= num_examples:
                break
        
        return examples
    
    def generate_terms_examples(self, num_examples: int = 25) -> List[TrainingExample]:
        """Generate financial terms explanation examples."""
        examples = []
        
        # Use definitions from corpus
        if self.definitions:
            term_groups = [
                ['Earnings Per Share (EPS)', 'Price-to-Earnings Ratio (P/E Ratio)', 'Market Capitalization (Market Cap)'],
                ['Revenue', 'Net Income', 'Gross Margin'],
                ['Risk Factor', 'Regulatory Risk', 'Concentration Risk'],
                ['10-K Filing', '10-Q Filing', 'MD&A (Management Discussion and Analysis)'],
                ['Free Cash Flow', 'Operating Cash Flow', 'Debt-to-Equity Ratio']
            ]
            
            for group in term_groups:
                terms_text = []
                for term_name in group:
                    term = self.definitions.get_term(term_name)
                    if term:
                        terms_text.append(f"{term_name}: {term.definition[:100]}...")
                
                if not terms_text:
                    continue
                
                input_text = f"""Context: A student investor encountered these terms:
{chr(10).join(terms_text)}

Query: {self.SECTION_TEMPLATES['terms']['query']}"""
                
                output_parts = ["6. Key Terms Explained\n"]
                for term_name in group:
                    term = self.definitions.get_term(term_name)
                    if term:
                        output_parts.append(f"• **{term.term}**: {term.definition}")
                        if term.example:
                            output_parts.append(f"  _Example: {term.example}_")
                
                output_parts.append("\nSources: Investor.gov, Investopedia")
                
                examples.append(TrainingExample(
                    input=input_text,
                    output="\n".join(output_parts),
                    section_type='terms',
                    ticker=''
                ))
                
                if len(examples) >= num_examples:
                    break
        
        return examples
    
    def generate_takeaway_examples(self, num_examples: int = 20) -> List[TrainingExample]:
        """Generate investor takeaway examples."""
        examples = []
        
        takeaway_templates = [
            "is financially strong, but its valuation is already high. Future returns depend on continued growth exceeding market expectations.",
            "offers steady growth with manageable risks. It may be suitable for investors seeking stability rather than aggressive growth.",
            "is experiencing rapid growth but comes with higher volatility. Beginners should understand that fast-growing companies can have large price swings.",
            "has strong competitive advantages but faces regulatory uncertainties. Students should monitor how new regulations might affect the business."
        ]
        
        for i, (ticker, info) in enumerate(self.COMPANY_INFO.items()):
            if i >= num_examples:
                break
            
            template = takeaway_templates[i % len(takeaway_templates)]
            
            input_text = f"""Context: Analysis of {info['name']} shows:
- Strong market position in {info['description'].split()[0:5]}
- Key risks: {', '.join(info['risks'][:2])}
- Opportunities: {', '.join(info['opportunities'][:2])}

Query: {self.SECTION_TEMPLATES['takeaway']['query'].format(company=info['name'])}"""
            
            output_text = f"""5. What This Means for a New Investor

{info['name']} {template}

Remember: Strong companies don't always make the best investments at any price. 
Consider your investment goals, time horizon, and risk tolerance.

⚠️ This is educational content, not investment advice."""
            
            examples.append(TrainingExample(
                input=input_text,
                output=output_text,
                section_type='takeaway',
                ticker=ticker
            ))
        
        return examples
    
    def build_training_set(self, target_size: int = 150) -> List[Dict]:
        """
        Build a complete training set.
        
        Args:
            target_size: Target number of examples (default 150)
            
        Returns:
            List of training examples as dictionaries
        """
        print("Building training data...")
        
        # Calculate distribution
        per_section = target_size // 6
        
        all_examples = []
        
        # Generate examples for each section type
        all_examples.extend(self.generate_summary_examples(per_section))
        print(f"  Generated {len(all_examples)} summary examples")
        
        all_examples.extend(self.generate_risk_examples(per_section + 5))
        print(f"  Generated {len(all_examples)} total (added risks)")
        
        all_examples.extend(self.generate_opportunity_examples(per_section))
        print(f"  Generated {len(all_examples)} total (added opportunities)")
        
        all_examples.extend(self.generate_metrics_examples(per_section))
        print(f"  Generated {len(all_examples)} total (added metrics)")
        
        all_examples.extend(self.generate_terms_examples(per_section))
        print(f"  Generated {len(all_examples)} total (added terms)")
        
        all_examples.extend(self.generate_takeaway_examples(per_section))
        print(f"  Generated {len(all_examples)} total (added takeaways)")
        
        # Shuffle
        random.shuffle(all_examples)
        
        # Convert to dicts
        result = [ex.to_dict() for ex in all_examples]
        
        print(f"Built {len(result)} training examples")
        return result
    
    def save_training_data(self, output_path: str, target_size: int = 150):
        """
        Build and save training data to JSON file.
        
        Args:
            output_path: Path to save the training data
            target_size: Target number of examples
        """
        examples = self.build_training_set(target_size)
        
        # Split into train/eval (90/10)
        split_idx = int(len(examples) * 0.9)
        train_data = examples[:split_idx]
        eval_data = examples[split_idx:]
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save training data
        output = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'total_examples': len(examples),
                'train_examples': len(train_data),
                'eval_examples': len(eval_data),
                'sections': list(set(ex['section_type'] for ex in examples))
            },
            'train': train_data,
            'eval': eval_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved training data to {output_path}")
        print(f"  Train: {len(train_data)} examples")
        print(f"  Eval: {len(eval_data)} examples")
        
        return output


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FinBrief training data")
    parser.add_argument('--output', '-o', type=str, default='data/training/finbrief_training.json',
                       help='Output path for training data')
    parser.add_argument('--size', '-s', type=int, default=150,
                       help='Target number of examples')
    parser.add_argument('--preview', action='store_true',
                       help='Preview examples without saving')
    
    args = parser.parse_args()
    
    builder = TrainingDataBuilder()
    
    if args.preview:
        examples = builder.build_training_set(10)
        print("\n=== Sample Training Examples ===\n")
        for i, ex in enumerate(examples[:3]):
            print(f"--- Example {i+1} ({ex['section_type']}) ---")
            print(f"INPUT:\n{ex['input'][:200]}...")
            print(f"\nOUTPUT:\n{ex['output'][:200]}...")
            print()
    else:
        builder.save_training_data(args.output, args.size)

