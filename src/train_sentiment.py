"""
Train sentiment classifier on FinancialPhraseBank dataset.
Generates visualizations and evaluation metrics.
"""
import os
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm


class FinancialPhraseBankDataset(Dataset):
    """Dataset for FinancialPhraseBank sentences."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_financial_phrase_bank(file_path: str):
    """
    Load FinancialPhraseBank dataset.
    
    Args:
        file_path: Path to Sentences_XXAgree.txt file
        
    Returns:
        texts: List of sentences
        labels: List of labels (0=negative, 1=neutral, 2=positive)
    """
    texts = []
    labels = []
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    print(f"Loading FinancialPhraseBank from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Format: "sentence@label"
            if '@' in line:
                parts = line.rsplit('@', 1)
                if len(parts) == 2:
                    text, label_str = parts
                    label_str = label_str.strip().lower()
                    
                    if label_str in label_map:
                        texts.append(text.strip())
                        labels.append(label_map[label_str])
    
    print(f"Loaded {len(texts)} sentences")
    
    # Print label distribution
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: 'negative', 1: 'neutral', 2: 'positive'}
    print("\nLabel distribution:")
    for label_id, count in zip(unique, counts):
        pct = (count / len(labels)) * 100
        print(f"  {label_names[label_id]}: {count} ({pct:.1f}%)")
    
    return texts, labels


def plot_training_curves(train_losses, val_losses, output_dir):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved loss_curve.png")


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Negative', 'Neutral', 'Positive']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion_matrix.png")


def plot_classification_report(y_true, y_pred, output_dir):
    """Plot classification metrics as bar chart."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2]
    )
    
    labels = ['Negative', 'Neutral', 'Positive']
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_report.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved classification_report.png")


def plot_roc_curves(y_true, y_pred_probs, output_dir):
    """Plot ROC curves for each class (one-vs-rest)."""
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = 3
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    labels = ['Negative', 'Neutral', 'Positive']
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    
    for i, color, label in zip(range(n_classes), colors, labels):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{label} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved roc_curves.png")


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy, all_preds, all_labels, np.array(all_probs)


def main():
    """Main training function."""
    # Configuration
    dataset_path = "/Users/arnavmeduri/projects/duke-ai-gateway/FinancialPhraseBank-v1.0/Sentences_75Agree.txt"
    output_dir = "/Users/arnavmeduri/projects/CS372/models/sentiment_classifier"
    model_dir = os.path.join(output_dir, "model")
    plots_dir = os.path.join(output_dir, "training_plots")
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Hyperparameters
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 4
    max_length = 512
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load dataset
    texts, labels = load_financial_phrase_bank(dataset_path)
    
    # Split into train/val/test (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples\n")
    
    # Initialize tokenizer and model
    print("Loading DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3
    )
    model.to(device)
    
    # Create datasets
    train_dataset = FinancialPhraseBankDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = FinancialPhraseBankDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = FinancialPhraseBankDataset(X_test, y_test, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...\n")
    train_losses = []
    val_losses = []
    val_accuracies = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds\n")
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}\n")
    
    # Classification report
    report = classification_report(
        test_labels, test_preds,
        target_names=['Negative', 'Neutral', 'Positive'],
        digits=4
    )
    print("Classification Report:")
    print(report)
    
    # Save metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, labels=[0, 1, 2], average=None
    )
    
    metrics = {
        "model": "distilbert-base-uncased",
        "dataset": "FinancialPhraseBank-75Agree",
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_time_seconds": training_time,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "per_class_metrics": {
            "negative": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1_score": float(f1[0]),
                "support": int(support[0])
            },
            "neutral": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1_score": float(f1[1]),
                "support": int(support[1])
            },
            "positive": {
                "precision": float(precision[2]),
                "recall": float(recall[2]),
                "f1_score": float(f1[2]),
                "support": int(support[2])
            }
        },
        "macro_avg": {
            "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)),
            "f1_score": float(np.mean(f1))
        },
        "weighted_avg": {
            "precision": float(precision_recall_fscore_support(test_labels, test_preds, average='weighted')[0]),
            "recall": float(precision_recall_fscore_support(test_labels, test_preds, average='weighted')[1]),
            "f1_score": float(precision_recall_fscore_support(test_labels, test_preds, average='weighted')[2])
        }
    }
    
    metrics_path = os.path.join(output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_curves(train_losses, val_losses, plots_dir)
    plot_confusion_matrix(test_labels, test_preds, plots_dir)
    plot_classification_report(test_labels, test_preds, plots_dir)
    plot_roc_curves(test_labels, test_probs, plots_dir)
    
    # Save model
    print(f"\nSaving model to {model_dir}...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    print("\nTraining complete!")
    print(f"Model saved to: {model_dir}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()

