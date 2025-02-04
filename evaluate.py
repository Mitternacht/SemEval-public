import torch
import pandas as pd
from model import ImprovedEmotionClassifier
from transformers import RobertaTokenizer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from data import EmotionDataModule
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch.nn.functional as F

def find_best_checkpoint(checkpoint_dir='checkpoints') -> str:
    """Find the checkpoint with the highest validation F1 score."""
    checkpoints = list(Path(checkpoint_dir).glob('emotion-*.ckpt'))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    checkpoint_scores = []
    for ckpt in checkpoints:
        try:
            f1_score = float(str(ckpt).split('val_f1=')[1].split('.ckpt')[0])
            checkpoint_scores.append((f1_score, ckpt))
        except:
            continue
    
    if not checkpoint_scores:
        raise ValueError("No valid checkpoints found")
    
    best_score, best_checkpoint = max(checkpoint_scores)
    print(f"Using checkpoint: {best_checkpoint} (F1: {best_score:.4f})")
    return str(best_checkpoint)

def analyze_predictions(model: ImprovedEmotionClassifier, 
                     data_module: EmotionDataModule,
                     threshold: float = 0.5) -> Dict:
    """Detailed analysis of model predictions."""
    model.eval()
    device = model.device
    all_preds = []
    all_labels = []
    all_logits = []
    emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
    
    # Get predictions
    val_loader = data_module.val_dataloader()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            logits = model(input_ids, attention_mask)
            preds = torch.sigmoid(logits)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_logits.append(logits.cpu())
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    
    # Convert to binary predictions
    binary_preds = (all_preds > threshold).float()
    
    # Create output directory
    os.makedirs('analysis', exist_ok=True)
    
    # Analyze per emotion
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    for idx, emotion in enumerate(emotion_labels):
        # Confusion matrix
        cm = confusion_matrix(all_labels[:, idx], binary_preds[:, idx])
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store results
        results[emotion] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        }
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
        axes[idx].set_title(f'{emotion.capitalize()} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('analysis/confusion_matrices.png')
    plt.close()
    
    # Threshold analysis
    analyze_thresholds(all_preds, all_labels, emotion_labels)
    
    # Calibration analysis
    analyze_calibration(all_preds, all_labels, emotion_labels)
    
    # Error analysis
    analyze_errors(all_preds, all_labels, all_logits, emotion_labels)
    
    return results

def analyze_thresholds(preds: torch.Tensor, 
                      labels: torch.Tensor,
                      emotion_labels: List[str],
                      thresholds: np.ndarray = np.arange(0.1, 1.0, 0.05)):
    """Analyze the effect of different prediction thresholds."""
    threshold_results = {emotion: {'precision': [], 'recall': [], 'f1': []} 
                        for emotion in emotion_labels}
    
    for thresh in thresholds:
        binary_preds = (preds > thresh).float()
        for idx, emotion in enumerate(emotion_labels):
            cm = confusion_matrix(labels[:, idx], binary_preds[:, idx])
            tn, fp, fn, tp = cm.ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            threshold_results[emotion]['precision'].append(precision)
            threshold_results[emotion]['recall'].append(recall)
            threshold_results[emotion]['f1'].append(f1)
    
    # Plot threshold analysis
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    for idx, emotion in enumerate(emotion_labels):
        axes[idx].plot(thresholds, threshold_results[emotion]['precision'], label='Precision')
        axes[idx].plot(thresholds, threshold_results[emotion]['recall'], label='Recall')
        axes[idx].plot(thresholds, threshold_results[emotion]['f1'], label='F1')
        axes[idx].set_title(f'{emotion.capitalize()} Metrics vs Threshold')
        axes[idx].set_xlabel('Threshold')
        axes[idx].set_ylabel('Score')
        axes[idx].legend()
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig('analysis/threshold_analysis.png')
    plt.close()

def analyze_calibration(preds: torch.Tensor,
                       labels: torch.Tensor,
                       emotion_labels: List[str],
                       n_bins: int = 10):
    """Analyze model calibration."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    for idx, emotion in enumerate(emotion_labels):
        pred_probs = preds[:, idx].numpy()
        true_labels = labels[:, idx].numpy()
        
        # Calculate calibration curve
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (pred_probs >= low) & (pred_probs < high)
            if mask.sum() > 0:
                bin_acc = true_labels[mask].mean()
                bin_conf = pred_probs[mask].mean()
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                bin_counts.append(mask.sum())
        
        # Plot calibration
        axes[idx].plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
        axes[idx].plot(bin_confidences, bin_accuracies, 'b-', label='Model')
        
        # Add histogram of predictions
        ax2 = axes[idx].twinx()
        ax2.hist(pred_probs, bins=n_bins, alpha=0.3, color='gray')
        
        axes[idx].set_title(f'{emotion.capitalize()} Calibration')
        axes[idx].set_xlabel('Predicted Probability')
        axes[idx].set_ylabel('True Probability')
        axes[idx].legend()
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig('analysis/calibration.png')
    plt.close()

def analyze_errors(preds: torch.Tensor,
                  labels: torch.Tensor,
                  logits: torch.Tensor,
                  emotion_labels: List[str]):
    """Analyze error patterns."""
    binary_preds = (preds > 0.5).float()
    
    # Analyze co-occurrence in errors
    error_mask = (binary_preds != labels)
    error_cooccurrence = torch.zeros((len(emotion_labels), len(emotion_labels)))
    
    for i in range(len(emotion_labels)):
        for j in range(len(emotion_labels)):
            error_cooccurrence[i, j] = ((error_mask[:, i] == 1) & 
                                      (error_mask[:, j] == 1)).sum()
    
    # Plot error co-occurrence
    plt.figure(figsize=(10, 8))
    sns.heatmap(error_cooccurrence.numpy(),
                annot=True,
                fmt='d',
                xticklabels=emotion_labels,
                yticklabels=emotion_labels)
    plt.title('Error Co-occurrence Matrix')
    plt.tight_layout()
    plt.savefig('analysis/error_cooccurrence.png')
    plt.close()
    
    # Analyze confidence distribution in errors vs correct predictions
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    for idx, emotion in enumerate(emotion_labels):
        correct_conf = preds[binary_preds[:, idx] == labels[:, idx], idx]
        error_conf = preds[binary_preds[:, idx] != labels[:, idx], idx]
        
        axes[idx].hist([correct_conf, error_conf], 
                      label=['Correct', 'Error'],
                      bins=20,
                      alpha=0.6)
        axes[idx].set_title(f'{emotion.capitalize()} Confidence Distribution')
        axes[idx].set_xlabel('Confidence')
        axes[idx].set_ylabel('Count')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig('analysis/confidence_distribution.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate emotion detection model')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to specific checkpoint. If not provided, will use best checkpoint.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory containing checkpoints')
    parser.add_argument('--data_dir', type=str, default='data/semeval',
                      help='Directory containing data files')
    args = parser.parse_args()

    # Find checkpoint to use
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = find_best_checkpoint(args.checkpoint_dir)

    # Setup data module
    data_module = EmotionDataModule(
        train_file=f'{args.data_dir}/train/eng.csv',
        val_file=f'{args.data_dir}/dev/eng.csv',
        test_file=f'{args.data_dir}/test/eng.csv',
        batch_size=16
    )
    data_module.setup()

    # Load model
    model = ImprovedEmotionClassifier.load_from_checkpoint(
        checkpoint_path, 
        map_location='cpu'
    )
    
    if torch.backends.mps.is_available():
        model = model.to('mps')
    
    # Run analysis
    results = analyze_predictions(model, data_module)
    
    # Print summary
    print("\nDetailed Analysis Results:")
    print("-" * 50)
    
    for emotion, metrics in results.items():
        print(f"\n{emotion.upper()}:")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"True Negatives: {metrics['true_negatives']}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()