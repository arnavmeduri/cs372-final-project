"""
This file loads pre-trained model weights from fine-tuning DistilBERT model on the Financial PhraseBank dataset.
I conducted fine-tuning on Google Colab with GPU (results documented in notebooks/sentiment_analysis_training).
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict

class SentimentClassifier:
    """
    Sentiment classifier using fine-tuned DistilBERT model.
    """
    
    def __init__(self, model_path: str = "models"):
        """Initialize sentiment classifier."""
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        

        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        self.label2id = {}
        for label_id, label_name in self.id2label.items():
            self.label2id[label_name] = label_id
        
    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        if self.model is None:
            print(f"loading sentiment model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=3
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"sentiment model loaded on {self.device}")
    
    def classify_text(self, text: str) -> Dict[str, any]:
        """Classify sentiment of a single text."""
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
            
            # get probabilities
            all_probs = torch.softmax(logits, dim=-1)
            probs = all_probs[0]
            
            # Predicted label
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
        """Classify sentiment of multiple texts efficiently."""
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
    # Testing it out
    classifier = SentimentClassifier()
    
    test_sentences = [
        "The company reported strong revenue growth and exceeded expectations.",
        "Revenues declined significantly due to market headwinds.",
        "The quarterly earnings were released on schedule."
    ]
    
    print("\nSingle classification:")
    result = classifier.classify_text(test_sentences[0])
    print(f"text: {test_sentences[0]}")
    print(f"sentiment: {result['label']}")
    print(f"scores: {result['scores']}\n")
    
    print("\nBatch classification:")
    results = classifier.classify_batch(test_sentences)
    for i in range(len(test_sentences)):
        text = test_sentences[i]
        result = results[i]
        print(f"text: {text}")
        print(f"sentiment: {result['label']}")
        print(f"scores: {result['scores']}\n")
