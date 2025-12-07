#!/usr/bin/env python3
"""
Test script for Duke AI Gateway integration.

Usage:
    python scripts/test_duke_gateway.py
    python scripts/test_duke_gateway.py --model "gpt-5"
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.duke_gateway_model import DukeGatewayModel


def test_duke_gateway(model_name: str = "GPT 4.1"):
    """Test Duke Gateway with a simple query."""
    print(f"\n{'='*80}")
    print(f"Testing Duke AI Gateway with model: {model_name}")
    print(f"{'='*80}\n")
    
    try:
        # Initialize model
        model = DukeGatewayModel(model_name=model_name, verbose=True)
        
        # Test 1: Simple question
        print("\n[Test 1] Simple Question")
        print("-" * 80)
        result = model.generate(
            prompt="What is a stock market?",
            max_new_tokens=100
        )
        print(f"Response: {result}\n")
        
        # Test 2: Context-based analysis
        print("\n[Test 2] Context-Based Analysis")
        print("-" * 80)
        context = """
        Apple Inc. designs and manufactures consumer electronics including iPhones, iPads, and Mac computers.
        The company reported revenue of $394.3 billion in 2023.
        Apple's P/E ratio is 36.8, indicating investors expect strong future growth.
        """
        result = model.analyze_with_context(
            context=context,
            query="Explain what Apple does in simple terms for a beginner investor.",
            for_beginners=True
        )
        print(f"Response: {result}\n")
        
        print("✅ All tests passed!")
        return True
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nTroubleshooting:")
        print("1. Set LITELLM_TOKEN in your .env file")
        print("2. Get your token from: https://dashboard.ai.duke.edu/")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_models():
    """List available models."""
    print("\nAvailable Duke AI Gateway Models:")
    print("-" * 80)
    models = DukeGatewayModel.list_available_models()
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Duke AI Gateway")
    parser.add_argument('--model', type=str, default='GPT 4.1',
                       help='Model to test (default: GPT 4.1)')
    parser.add_argument('--list', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    else:
        success = test_duke_gateway(args.model)
        sys.exit(0 if success else 1)


