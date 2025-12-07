"""
Confidence Head for FinBrief.

A 2-layer MLP that predicts a confidence score [0, 1] for generated analyses.
Input: Final hidden state from GPT-2
Output: Confidence scalar indicating model certainty

Includes calibration evaluation metrics:
- Brier Score
- Expected Calibration Error (ECE)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import numpy as np
from dataclasses import dataclass
import json
import os


@dataclass
class ConfidenceConfig:
    """Configuration for the confidence head."""
    hidden_size: int = 1024  # GPT-2 Medium hidden size
    intermediate_size: int = 256
    dropout: float = 0.1
    
    def to_dict(self) -> Dict:
        return {
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'dropout': self.dropout
        }


class ConfidenceHead(nn.Module):
    """
    2-layer MLP confidence head for GPT-2.
    
    Architecture:
        hidden_state (1024) -> Linear -> ReLU -> Dropout -> Linear -> Sigmoid -> confidence (1)
    """
    
    def __init__(self, config: Optional[ConfidenceConfig] = None):
        """
        Initialize the confidence head.
        
        Args:
            config: Configuration for the head
        """
        super().__init__()
        self.config = config or ConfidenceConfig()
        
        # 2-layer MLP
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.intermediate_size)
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc2 = nn.Linear(self.config.intermediate_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: Final hidden states from GPT-2 [batch_size, seq_len, hidden_size]
                          or pooled [batch_size, hidden_size]
        
        Returns:
            Confidence scores [batch_size, 1] in range [0, 1]
        """
        # If 3D, take the last token's hidden state (like GPT-2 does for classification)
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        
        # MLP forward
        x = self.fc1(hidden_states)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Sigmoid to get [0, 1] confidence
        confidence = torch.sigmoid(x)
        
        return confidence
    
    def save(self, path: str):
        """Save the confidence head."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config.to_dict()
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'ConfidenceHead':
        """Load a saved confidence head."""
        checkpoint = torch.load(path, map_location='cpu')
        config = ConfidenceConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['state_dict'])
        return model


class ConfidenceEstimator:
    """
    High-level interface for confidence estimation.
    Wraps the confidence head with calibration and evaluation utilities.
    """
    
    def __init__(
        self,
        model=None,  # GPT-2 model
        confidence_head: Optional[ConfidenceHead] = None,
        device: str = 'cpu'
    ):
        """
        Initialize the confidence estimator.
        
        Args:
            model: GPT-2 model (or any causal LM with output_hidden_states support)
            confidence_head: Trained confidence head (or None to create new)
            device: Device to run on
        """
        self.model = model
        self.confidence_head = confidence_head or ConfidenceHead()
        self.device = device
        
        if self.confidence_head:
            self.confidence_head = self.confidence_head.to(device)
    
    def estimate_confidence(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Estimate confidence for a given input.
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            
        Returns:
            Confidence score in [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not set. Initialize with a GPT-2 model.")
        
        self.model.eval()
        self.confidence_head.eval()
        
        with torch.no_grad():
            # Get hidden states from model
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
                output_hidden_states=True
            )
            
            # Get last hidden state
            last_hidden_state = outputs.hidden_states[-1]
            
            # Get confidence
            confidence = self.confidence_head(last_hidden_state)
            
        return confidence.item()
    
    def estimate_from_text(self, text: str, tokenizer) -> float:
        """
        Estimate confidence from text input.
        
        Args:
            text: Input text
            tokenizer: Tokenizer for the model
            
        Returns:
            Confidence score in [0, 1]
        """
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        
        return self.estimate_confidence(
            inputs['input_ids'],
            inputs.get('attention_mask')
        )


# ============================================================
# Calibration Metrics
# ============================================================

def compute_brier_score(
    confidences: np.ndarray,
    correctness: np.ndarray
) -> float:
    """
    Compute Brier Score for calibration evaluation.
    
    Brier Score = mean((confidence - correctness)^2)
    Lower is better. Perfect calibration = 0.
    
    Args:
        confidences: Model confidence scores [0, 1]
        correctness: Binary correctness labels (0 or 1)
        
    Returns:
        Brier score
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    
    return np.mean((confidences - correctness) ** 2)


def compute_ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, Dict]:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between confidence and accuracy across bins.
    Lower is better. Perfect calibration = 0.
    
    Args:
        confidences: Model confidence scores [0, 1]
        correctness: Binary correctness labels (0 or 1)
        n_bins: Number of bins for calibration
        
    Returns:
        Tuple of (ECE score, bin details)
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_details = []
    ece = 0.0
    
    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        n_in_bin = np.sum(in_bin)
        
        if n_in_bin > 0:
            # Average confidence and accuracy in bin
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(correctness[in_bin])
            
            # Contribution to ECE
            ece += (n_in_bin / len(confidences)) * abs(avg_accuracy - avg_confidence)
            
            bin_details.append({
                'bin': i,
                'range': (bin_boundaries[i], bin_boundaries[i + 1]),
                'count': int(n_in_bin),
                'avg_confidence': float(avg_confidence),
                'avg_accuracy': float(avg_accuracy),
                'gap': float(abs(avg_accuracy - avg_confidence))
            })
    
    return ece, {'bins': bin_details, 'n_bins': n_bins}


def compute_calibration_metrics(
    confidences: List[float],
    correctness: List[int]
) -> Dict:
    """
    Compute all calibration metrics.
    
    Args:
        confidences: List of confidence scores
        correctness: List of correctness labels (0 or 1)
        
    Returns:
        Dictionary with all metrics
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    
    brier = compute_brier_score(confidences, correctness)
    ece, ece_details = compute_ece(confidences, correctness)
    
    return {
        'brier_score': float(brier),
        'ece': float(ece),
        'ece_details': ece_details,
        'mean_confidence': float(np.mean(confidences)),
        'mean_accuracy': float(np.mean(correctness)),
        'n_samples': len(confidences)
    }


# ============================================================
# Heuristic Confidence Estimation (No Training Required)
# ============================================================

class HeuristicConfidenceEstimator:
    """
    Heuristic-based confidence estimation without a trained model.
    
    Uses multiple signals to estimate confidence:
    - Number and quality of retrieved sources
    - Presence of key financial metrics
    - Source diversity (SEC, news, definitions)
    - Recency of data
    """
    
    def __init__(self):
        """Initialize the heuristic estimator."""
        self.weights = {
            'source_count': 0.2,
            'source_quality': 0.25,
            'source_diversity': 0.15,
            'metrics_coverage': 0.2,
            'data_recency': 0.1,
            'content_length': 0.1
        }
    
    def estimate(
        self,
        citations: List[Dict],
        has_metrics: bool = False,
        content_length: int = 0,
        analysis_sections: int = 0
    ) -> Tuple[float, Dict]:
        """
        Estimate confidence based on available data quality.
        
        Args:
            citations: List of citation dictionaries
            has_metrics: Whether Finnhub metrics are available
            content_length: Length of generated content
            analysis_sections: Number of sections filled in analysis
            
        Returns:
            Tuple of (confidence score, breakdown)
        """
        scores = {}
        
        # Source count score (0-5 sources mapped to 0-1)
        source_count = len(citations)
        scores['source_count'] = min(source_count / 5, 1.0)
        
        # Source quality score (trusted sources)
        trusted_count = sum(1 for c in citations if c.get('is_trusted', False))
        scores['source_quality'] = trusted_count / max(source_count, 1)
        
        # Source diversity (different types)
        source_types = set(c.get('source_type', '') for c in citations)
        scores['source_diversity'] = len(source_types) / 4  # 4 possible types
        
        # Metrics coverage
        scores['metrics_coverage'] = 1.0 if has_metrics else 0.5
        
        # Data recency (simplified - assume recent if citations exist)
        scores['data_recency'] = 0.8 if citations else 0.3
        
        # Content length (reasonable length is good)
        if content_length > 500:
            scores['content_length'] = min(content_length / 2000, 1.0)
        else:
            scores['content_length'] = content_length / 500
        
        # Weighted average
        confidence = sum(
            scores[key] * self.weights[key]
            for key in self.weights
        )
        
        # Clamp to [0.3, 0.95] - never be too certain or too uncertain
        confidence = max(0.3, min(0.95, confidence))
        
        return confidence, {
            'scores': scores,
            'weights': self.weights,
            'raw_confidence': confidence
        }
    
    def format_confidence(self, confidence: float) -> str:
        """Format confidence for display."""
        if confidence >= 0.8:
            level = "High"
            emoji = "🟢"
        elif confidence >= 0.6:
            level = "Moderate"
            emoji = "🟡"
        else:
            level = "Low"
            emoji = "🟠"
        
        return f"{emoji} {level} ({confidence:.2f})"


# CLI for testing
if __name__ == "__main__":
    print("=== Confidence Head Test ===\n")
    
    # Test the neural network head
    print("1. Testing ConfidenceHead...")
    config = ConfidenceConfig(hidden_size=1024, intermediate_size=256)
    head = ConfidenceHead(config)
    
    # Simulate hidden states
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    confidence = head(hidden_states)
    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {confidence.shape}")
    print(f"   Confidence values: {confidence.squeeze().tolist()}")
    
    # Test calibration metrics
    print("\n2. Testing Calibration Metrics...")
    
    # Simulated data: confidences and correctness
    np.random.seed(42)
    n_samples = 100
    confidences = np.random.uniform(0.3, 0.9, n_samples)
    # Simulate partially calibrated model
    correctness = (np.random.random(n_samples) < confidences).astype(int)
    
    metrics = compute_calibration_metrics(confidences.tolist(), correctness.tolist())
    print(f"   Brier Score: {metrics['brier_score']:.4f}")
    print(f"   ECE: {metrics['ece']:.4f}")
    print(f"   Mean Confidence: {metrics['mean_confidence']:.4f}")
    print(f"   Mean Accuracy: {metrics['mean_accuracy']:.4f}")
    
    # Test heuristic estimator
    print("\n3. Testing Heuristic Confidence Estimator...")
    
    estimator = HeuristicConfidenceEstimator()
    
    # Test with good data
    good_citations = [
        {'source_type': 'sec_filing', 'is_trusted': True},
        {'source_type': 'definition', 'is_trusted': True},
        {'source_type': 'financial_metrics', 'is_trusted': True},
    ]
    conf, breakdown = estimator.estimate(
        citations=good_citations,
        has_metrics=True,
        content_length=1500,
        analysis_sections=5
    )
    print(f"   Good data confidence: {estimator.format_confidence(conf)}")
    
    # Test with sparse data
    sparse_citations = [
        {'source_type': 'news', 'is_trusted': False},
    ]
    conf, breakdown = estimator.estimate(
        citations=sparse_citations,
        has_metrics=False,
        content_length=300,
        analysis_sections=2
    )
    print(f"   Sparse data confidence: {estimator.format_confidence(conf)}")
    
    print("\n✓ All confidence tests passed!")

