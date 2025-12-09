"""
Sentiment Classifier for Financial Text
Uses fine-tuned DistilBERT model to classify sentiment of SEC filing chunks.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict

class SentimentClassifier:
    """
    Sentiment classifier using fine-tuned DistilBERT model.
    Trained on Financial PhraseBank dataset.
    """
    
    def __init__(self, model_path: str = "models/distillbert-fine-tuning"):
        """
        Initialize sentiment classifier.
        
        Args:
            model_path: Path to fine-tuned model directory
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        
        # Label mapping from training (0=negative, 1=neutral, 2=positive)
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        
    def _get_device(self) -> str:
        """Determine best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Lazy load model and tokenizer."""
        if self.model is None:
            print(f"[SENTIMENT] Loading model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=3
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"[SENTIMENT] Model loaded on {self.device}")
    
    def classify_text(self, text: str) -> Dict[str, any]:
        """
        Classify sentiment of a single text.
        
        Args:
            text: Text to classify
            
        Returns:
            Dict with 'label' and 'scores' for each sentiment
        """
        self._load_model()
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = torch.argmax(logits, dim=-1).item()
        
        return {
            "label": self.id2label[pred_id],
            "scores": {
                "negative": probs[0].item(),
                "neutral": probs[1].item(),
                "positive": probs[2].item()
            }
        }
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Classify sentiment of multiple texts efficiently.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of dicts with 'label' and 'scores' for each text
        """
        self._load_model()
        
        if not texts:
            return []
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(logits, dim=-1)
        
        results = []
        for i in range(len(texts)):
            results.append({
                "label": self.id2label[pred_ids[i].item()],
                "scores": {
                    "negative": probs[i][0].item(),
                    "neutral": probs[i][1].item(),
                    "positive": probs[i][2].item()
                }
            })
        
        return results


if __name__ == "__main__":
    # Example usage
    classifier = SentimentClassifier()
    
    test_sentences = [
        "The company reported strong revenue growth and exceeded expectations.",
        "Revenues declined significantly due to market headwinds.",
        "The quarterly earnings were released on schedule."
    ]
    
    print("\n--- Single Classification ---")
    result = classifier.classify_text(test_sentences[0])
    print(f"Text: {test_sentences[0]}")
    print(f"Sentiment: {result['label']}")
    print(f"Scores: {result['scores']}\n")
    
    print("\n--- Batch Classification ---")
    results = classifier.classify_batch(test_sentences)
    for text, result in zip(test_sentences, results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']}")
        print(f"Scores: {result['scores']}\n")
