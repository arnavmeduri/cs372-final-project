"""
LoRA (Low-Rank Adaptation) trainer for FinBrief.

Fine-tunes GPT-2 Medium to generate structured, student-oriented financial briefs.
Uses PEFT library for efficient parameter-efficient fine-tuning.

Configuration (per design.md):
- Rank r = 16
- Alpha α = 32
- Dropout = 0.1
- Target: GPT-2 attention layers (c_attn, c_proj)
- Training: 3-5 epochs on 150-200 examples
"""
import os
import json
import torch
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from datasets import Dataset


@dataclass
class FinBriefLoRAConfig:
    """Configuration for FinBrief LoRA fine-tuning."""
    
    # Model configuration
    base_model: str = "gpt2-medium"
    
    # LoRA hyperparameters (per design.md)
    lora_r: int = 16  # Rank
    lora_alpha: int = 32  # Alpha scaling factor
    lora_dropout: float = 0.1
    
    # Target modules - auto-detect based on model architecture
    # GPT-2 uses: c_attn, c_proj
    # LLaMA/TinyLlama uses: q_proj, k_proj, v_proj, o_proj
    target_modules: List[str] = None  # Will be auto-detected if None
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 4
    
    # Paths
    output_dir: str = "models/lora_adapter"
    logging_dir: str = "logs/lora_training"
    
    # Evaluation
    eval_steps: int = 50
    save_steps: int = 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'FinBriefLoRAConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class FinBriefLoRATrainer:
    """
    Trainer for LoRA fine-tuning of GPT-2 for FinBrief output generation.
    """
    
    def __init__(self, config: Optional[FinBriefLoRAConfig] = None):
        """
        Initialize the trainer.
        
        Args:
            config: LoRA configuration (uses defaults if not provided)
        """
        self.config = config or FinBriefLoRAConfig()
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def setup(self):
        """Load base model and apply LoRA configuration."""
        print(f"Loading base model: {self.config.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float32,  # Use float32 for training stability
        )
        
        # Auto-detect target modules based on model architecture
        target_modules = self.config.target_modules
        if target_modules is None:
            model_name_lower = self.config.base_model.lower()
            if 'gpt2' in model_name_lower or 'distilgpt2' in model_name_lower:
                # GPT-2 architecture
                target_modules = ["c_attn", "c_proj"]
            elif 'llama' in model_name_lower or 'tinyllama' in model_name_lower:
                # LLaMA architecture
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                # Default to GPT-2 style (most common)
                target_modules = ["c_attn", "c_proj"]
                print(f"Warning: Unknown model architecture for {self.config.base_model}, using GPT-2 target modules")
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        # Apply LoRA to model
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        print("LoRA configuration applied successfully")
        return self
    
    def prepare_dataset(self, examples: List[Dict[str, str]]) -> Dataset:
        """
        Prepare dataset from training examples.
        
        Args:
            examples: List of dicts with 'input' and 'output' keys
            
        Returns:
            HuggingFace Dataset
        """
        # Format examples as prompt-completion pairs
        formatted_texts = []
        
        for ex in examples:
            # Create the full training text
            # Format: <context>\n\n<response>
            text = f"{ex['input']}\n\n{ex['output']}{self.tokenizer.eos_token}"
            formatted_texts.append(text)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding='max_length',
            )
        
        dataset = Dataset.from_dict({'text': formatted_texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        return tokenized_dataset
    
    def train(
        self,
        train_examples: List[Dict[str, str]],
        eval_examples: Optional[List[Dict[str, str]]] = None
    ):
        """
        Train the LoRA adapter.
        
        Args:
            train_examples: Training examples with 'input' and 'output' keys
            eval_examples: Optional evaluation examples
        """
        if self.peft_model is None:
            self.setup()
        
        print(f"Preparing {len(train_examples)} training examples...")
        train_dataset = self.prepare_dataset(train_examples)
        
        eval_dataset = None
        if eval_examples:
            print(f"Preparing {len(eval_examples)} evaluation examples...")
            eval_dataset = self.prepare_dataset(eval_examples)
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.logging_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_dir=self.config.logging_dir,
            logging_steps=10,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none",  # Disable wandb/tensorboard
            fp16=False,  # Use fp32 for stability
            dataloader_drop_last=True,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        print("Starting training...")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  LoRA rank: {self.config.lora_r}")
        print(f"  LoRA alpha: {self.config.lora_alpha}")
        
        # Train
        train_result = trainer.train()
        
        # Save the LoRA adapter
        print(f"Saving LoRA adapter to {self.config.output_dir}")
        self.peft_model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save config
        self.config.save(os.path.join(self.config.output_dir, "finbrief_config.json"))
        
        # Save training metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(os.path.join(self.config.output_dir, "training_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("Training complete!")
        return train_result
    
    def load_adapter(self, adapter_path: str):
        """
        Load a trained LoRA adapter.
        
        Args:
            adapter_path: Path to the saved adapter
        """
        print(f"Loading LoRA adapter from {adapter_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(self.config.base_model)
        
        # Load LoRA adapter
        self.peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        self.peft_model.eval()
        
        print("Adapter loaded successfully")
        return self
    
    def generate(self, prompt: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
        """
        Generate text using the fine-tuned model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if self.peft_model is None:
            raise ValueError("No model loaded. Call setup() or load_adapter() first.")
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length
        )
        
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from output
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        
        return generated


def create_sample_training_examples() -> List[Dict[str, str]]:
    """
    Create sample training examples for testing.
    In production, use training_data_builder.py to generate real examples.
    """
    examples = [
        {
            "input": """Context: Apple Inc reported revenue of $94.9 billion for Q3 2024, 
up 5% year over year. iPhone revenue was $46.2 billion. Services revenue reached 
$24.2 billion, a new all-time record.

Query: Provide a beginner-friendly company summary for Apple.""",
            "output": """1. Company Summary (Beginner-Friendly)

Apple Inc. designs and sells consumer electronics including the iPhone, Mac, iPad, 
and Apple Watch. The company also operates a growing Services business that includes 
the App Store, iCloud, Apple Music, and Apple TV+. Services have become an increasingly 
important part of Apple's revenue.

Citations: (10-Q Filing Q3 2024)"""
        },
        {
            "input": """Context: Risk Factors: The company faces intense competition in all 
of its markets. Global economic conditions could materially adversely affect the company. 
Supply chain disruptions could impact product availability.

Query: What are the key risks students should understand?""",
            "output": """4. Risks Students Should Understand

🟡 **Competition Risk** (Medium): Apple competes with many companies in every market 
it operates in. If competitors offer better products or lower prices, Apple could 
lose customers.

🟡 **Economic Risk** (Medium): When the economy struggles, people buy fewer expensive 
electronics. This could hurt Apple's sales.

🟠 **Supply Chain Risk** (Medium-High): Apple relies on factories in other countries 
to make its products. If something disrupts these factories, Apple might not have 
enough products to sell.

Citations: (10-K Item 1A: Risk Factors)"""
        },
        {
            "input": """Context: P/E Ratio: 28.5, Market Cap: $2.8T, Revenue Growth: +8%, 
EPS: $6.15, Debt-to-Equity: 1.5

Query: Explain these financial metrics for a beginner investor.""",
            "output": """2. Key Financial Metrics for Students

• Market Cap: $2.8T (This company is worth $2.8 trillion - one of the largest in the world)
• P/E Ratio: 28.5 (Investors pay $28.50 for every $1 of earnings)
• EPS: $6.15 (The company earns $6.15 per share)
• Revenue Growth: +8% (Sales grew 8% compared to last year)
• Debt-to-Equity: 1.5 (The company has moderate debt)

Student Interpretation:
The P/E ratio of 28.5 is above the market average (~20), which suggests investors 
expect continued growth. The 8% revenue growth is solid but not exceptional. 
The debt level is manageable for a company this size.

Source: Finnhub Metrics"""
        },
    ]
    
    return examples


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FinBrief LoRA Trainer")
    parser.add_argument('--test', action='store_true', help='Run with sample data')
    parser.add_argument('--config', type=str, help='Path to config JSON')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running LoRA trainer test with sample data...")
        
        config = FinBriefLoRAConfig(
            num_epochs=1,
            batch_size=2,
            output_dir="models/lora_test"
        )
        
        trainer = FinBriefLoRATrainer(config)
        trainer.setup()
        
        # Use sample examples
        examples = create_sample_training_examples()
        print(f"Created {len(examples)} sample training examples")
        
        # Just test dataset preparation (don't actually train)
        dataset = trainer.prepare_dataset(examples)
        print(f"Dataset prepared with {len(dataset)} examples")
        print("Test passed! Ready for training.")
    else:
        print("Use --test to run a test, or use finetune.py for full training")

