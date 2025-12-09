"""
Financial Sentiment Classifier using DistilBERT.
Trained on FinancialPhraseBank dataset for financial text sentiment analysis.
"""
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import Dict, List, Optional
import numpy as np


class FinancialSentimentClassifier:
    """
    Sentiment classifier for financial text using fine-tuned DistilBERT.
    
    Classes:
    - 0: negative
    - 1: neutral
    - 2: positive
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the sentiment classifier.
        
        Args:
            model_path: Path to saved model directory. If None, uses default.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        self.reverse_label_map = {"negative": 0, "neutral": 1, "positive": 2}
        
    def load_model(self, model_path: Optional[str] = None):
        """
        Load a pre-trained sentiment model.
        
        Args:
            model_path: Path to model directory. If None, uses self.model_path.
        """
        if model_path is None:
            model_path = self.model_path
            
        if model_path is None:
            # Default path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            model_path = os.path.join(project_root, 'models', 'sentiment_classifier', 'model')
        
        print(f"Loading sentiment model from {model_path}")
        
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Sentiment model loaded successfully")
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            print("Model may not be trained yet. Please run train_sentiment.py first.")
            raise
    
    def classify_text(self, text: str, return_probabilities: bool = True) -> Dict[str, float]:
        """
        Classify sentiment of a single text.
        
        Args:
            text: Input text to classify
            return_probabilities: If True, returns probabilities for all classes
            
        Returns:
            Dictionary with sentiment predictions and probabilities
        """
        if self.model is None:
            self.load_model()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
        
        result = {
            "predicted_label": self.label_map[predicted_class],
            "predicted_class": predicted_class
        }
        
        if return_probabilities:
            probs_np = probs.cpu().numpy()[0]
            result["probabilities"] = {
                "negative": float(probs_np[0]),
                "neutral": float(probs_np[1]),
                "positive": float(probs_np[2])
            }
        
        return result
    
    def classify_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """
        Classify sentiment of multiple texts efficiently.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of dictionaries with predictions for each text
        """
        if self.model is None:
            self.load_model()
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(probs, dim=-1)
            
            # Convert to results
            probs_np = probs.cpu().numpy()
            predicted_classes_np = predicted_classes.cpu().numpy()
            
            for j in range(len(batch)):
                result = {
                    "predicted_label": self.label_map[predicted_classes_np[j]],
                    "predicted_class": int(predicted_classes_np[j]),
                    "probabilities": {
                        "negative": float(probs_np[j][0]),
                        "neutral": float(probs_np[j][1]),
                        "positive": float(probs_np[j][2])
                    }
                }
                results.append(result)
        
        return results
    
    def analyze_aggregate_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze aggregate sentiment across multiple texts.
        
        Args:
            texts: List of text chunks to analyze
            
        Returns:
            Dictionary with aggregate statistics
        """
        if not texts:
            return {
                "positive_pct": 0.0,
                "neutral_pct": 0.0,
                "negative_pct": 0.0,
                "total_chunks": 0,
                "overall_tone": "Unknown"
            }
        
        # Classify all texts
        predictions = self.classify_batch(texts)
        
        # Count predictions
        label_counts = {"negative": 0, "neutral": 0, "positive": 0}
        for pred in predictions:
            label_counts[pred["predicted_label"]] += 1
        
        total = len(predictions)
        
        # Calculate percentages
        positive_pct = (label_counts["positive"] / total) * 100
        neutral_pct = (label_counts["neutral"] / total) * 100
        negative_pct = (label_counts["negative"] / total) * 100
        
        # Determine overall tone
        max_label = max(label_counts, key=label_counts.get)
        max_pct = max(positive_pct, neutral_pct, negative_pct)
        
        # If no clear majority (within 10% of each other), call it "Mixed"
        if max_pct < 45:
            overall_tone = "Mixed"
        else:
            overall_tone = max_label.capitalize()
        
        return {
            "positive_pct": positive_pct,
            "neutral_pct": neutral_pct,
            "negative_pct": negative_pct,
            "total_chunks": total,
            "overall_tone": overall_tone,
            "label_counts": label_counts
        }


# Singleton instance for lazy loading
_classifier_instance = None


def get_classifier() -> FinancialSentimentClassifier:
    """Get or create singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = FinancialSentimentClassifier()
    return _classifier_instance


if __name__ == "__main__":
    # Test the classifier
    print("Testing Financial Sentiment Classifier\n")
    
    classifier = FinancialSentimentClassifier()
    
    try:
        classifier.load_model()
        
        # Test examples
        test_texts = [
            "The company reported strong revenue growth and increased profitability.",
            "Revenue declined significantly and the company faces regulatory challenges.",
            "The company maintained stable operations with no major changes."
        ]
        
        print("Single text classification:")
        for text in test_texts:
            result = classifier.classify_text(text)
            print(f"\nText: {text}")
            print(f"Predicted: {result['predicted_label']}")
            print(f"Probabilities: {result['probabilities']}")
        
        print("\n" + "="*80)
        print("Batch classification:")
        results = classifier.classify_batch(test_texts)
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result['predicted_label']} (confidence: {max(result['probabilities'].values()):.2f})")
        
        print("\n" + "="*80)
        print("Aggregate sentiment analysis:")
        agg = classifier.analyze_aggregate_sentiment(test_texts)
        print(f"Positive: {agg['positive_pct']:.1f}%")
        print(f"Neutral: {agg['neutral_pct']:.1f}%")
        print(f"Negative: {agg['negative_pct']:.1f}%")
        print(f"Overall tone: {agg['overall_tone']}")
        
    except Exception as e:
        print(f"\nModel not found or not trained yet.")
        print(f"Please run: python src/train_sentiment.py")

