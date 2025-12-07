#!/usr/bin/env python3
"""
FinBrief Fine-Tuning Script

Fine-tunes GPT-2 Medium with LoRA adapters for structured financial brief generation.

Usage:
    # Generate training data and fine-tune
    python -m src.finetune --build-data --train
    
    # Fine-tune with existing data
    python -m src.finetune --train --data data/training/finbrief_training.json
    
    # Test the fine-tuned model
    python -m src.finetune --test --adapter models/lora_adapter
"""
import os
import json
import argparse
from datetime import datetime

from .lora_trainer import FinBriefLoRATrainer, FinBriefLoRAConfig
from .training_data_builder import TrainingDataBuilder


def build_training_data(output_path: str, num_examples: int = 150) -> dict:
    """Build training data from templates."""
    print("\n=== Building Training Data ===\n")
    
    builder = TrainingDataBuilder()
    data = builder.save_training_data(output_path, num_examples)
    
    return data


def load_training_data(data_path: str) -> tuple:
    """Load training data from JSON file."""
    print(f"\nLoading training data from {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    train_examples = data['train']
    eval_examples = data.get('eval', [])
    
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Eval: {len(eval_examples)} examples")
    
    return train_examples, eval_examples


def train_model(
    train_examples: list,
    eval_examples: list = None,
    config: FinBriefLoRAConfig = None
):
    """Train the LoRA adapter."""
    print("\n=== Starting LoRA Fine-Tuning ===\n")
    
    config = config or FinBriefLoRAConfig()
    trainer = FinBriefLoRATrainer(config)
    
    # Setup and train
    trainer.setup()
    result = trainer.train(train_examples, eval_examples)
    
    return trainer, result


def test_model(adapter_path: str, num_tests: int = 3, model_name: str = None):
    """Test the fine-tuned model with sample prompts."""
    print(f"\n=== Testing Fine-Tuned Model ===\n")
    print(f"Loading adapter from: {adapter_path}\n")
    
    # Load config to get the base model used during training
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if model_name is None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            adapter_config = json.load(f)
            model_name = adapter_config.get("base_model_name_or_path", "gpt2-medium")
            print(f"Detected base model: {model_name}")
    
    # Map back from full path to short name if needed
    model_name = model_name or "gpt2-medium"
    config = FinBriefLoRAConfig(base_model=model_name)
    trainer = FinBriefLoRATrainer(config)
    trainer.load_adapter(adapter_path)
    
    test_prompts = [
        """Context: Apple Inc reported Q3 2024 revenue of $85.8 billion. iPhone revenue was $42.3 billion. 
Services revenue reached $22.2 billion, growing 14% year over year.

Query: Provide a beginner-friendly company summary for Apple Inc.""",
        
        """Context: Risk Factors: The company faces intense competition from larger technology companies. 
Supply chain disruptions could materially affect product availability and costs.

Query: What are the key risks students should understand?""",
        
        """Context: P/E Ratio: 28.5, Market Cap: $2.8T, Revenue Growth: +8%, EPS: $6.15

Query: Explain these financial metrics for a beginner investor.""",
    ]
    
    for i, prompt in enumerate(test_prompts[:num_tests]):
        print(f"--- Test {i+1} ---")
        print(f"PROMPT: {prompt[:100]}...\n")
        
        response = trainer.generate(prompt, max_new_tokens=200)
        print(f"RESPONSE:\n{response}\n")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="FinBrief LoRA Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: build data and train
  python -m src.finetune --build-data --train
  
  # Use custom data path
  python -m src.finetune --train --data my_data.json
  
  # Customize training
  python -m src.finetune --train --epochs 5 --batch-size 8 --lr 1e-4
  
  # Test trained model
  python -m src.finetune --test --adapter models/lora_adapter
  
  # Preview training data without training
  python -m src.finetune --build-data --preview
"""
    )
    
    # Data arguments
    parser.add_argument('--build-data', action='store_true',
                       help='Build training data from templates')
    parser.add_argument('--data', type=str, default='data/training/finbrief_training.json',
                       help='Path to training data JSON')
    parser.add_argument('--num-examples', type=int, default=150,
                       help='Number of training examples to generate')
    parser.add_argument('--preview', action='store_true',
                       help='Preview data without training')
    
    # Training arguments
    parser.add_argument('--train', action='store_true',
                       help='Run training')
    parser.add_argument('--model', type=str, default='gpt2-medium',
                       choices=['gpt2', 'gpt2-medium', 'distilgpt2', 'tinyllama'],
                       help='Base model to fine-tune (distilgpt2 is fastest, tinyllama is instruction-tuned)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--lora-r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--output-dir', type=str, default='models/lora_adapter',
                       help='Output directory for trained adapter')
    
    # Test arguments
    parser.add_argument('--test', action='store_true',
                       help='Test a trained adapter')
    parser.add_argument('--adapter', type=str, default='models/lora_adapter',
                       help='Path to trained adapter for testing')
    
    args = parser.parse_args()
    
    # Build training data if requested
    if args.build_data:
        os.makedirs(os.path.dirname(args.data) if os.path.dirname(args.data) else '.', exist_ok=True)
        build_training_data(args.data, args.num_examples)
        
        if args.preview:
            print("\n=== Training Data Preview ===\n")
            with open(args.data, 'r') as f:
                data = json.load(f)
            
            for i, ex in enumerate(data['train'][:3]):
                print(f"--- Example {i+1} ({ex['section_type']}) ---")
                print(f"INPUT:\n{ex['input'][:300]}...")
                print(f"\nOUTPUT:\n{ex['output'][:300]}...")
                print()
            return
    
    # Train if requested
    if args.train:
        # Map model name to full HuggingFace path if needed
        model_map = {
            'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        }
        base_model = model_map.get(args.model, args.model)
        
        # Create config
        config = FinBriefLoRAConfig(
            base_model=base_model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            output_dir=args.output_dir,
        )
        
        # Load or build data
        if not os.path.exists(args.data):
            print(f"Training data not found at {args.data}")
            print("Building training data...")
            build_training_data(args.data, args.num_examples)
        
        train_examples, eval_examples = load_training_data(args.data)
        
        # Train
        trainer, result = train_model(train_examples, eval_examples, config)
        
        print("\n=== Training Complete ===")
        print(f"Adapter saved to: {args.output_dir}")
        print(f"Final loss: {result.training_loss:.4f}")
        
        # Auto-test after training
        print("\nRunning quick test...")
        # Use the full model path for testing
        test_model(args.output_dir, num_tests=1, model_name=base_model)
    
    # Test if requested
    if args.test and not args.train:
        if not os.path.exists(args.adapter):
            print(f"Adapter not found at {args.adapter}")
            print("Train a model first with: python -m src.finetune --build-data --train")
            return
        
        test_model(args.adapter)
    
    # Show help if no action specified
    if not (args.build_data or args.train or args.test):
        parser.print_help()


if __name__ == "__main__":
    main()

