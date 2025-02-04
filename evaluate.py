import torch
import pandas as pd
from model import EmotionClassifier
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

def find_best_checkpoint(checkpoint_dir='checkpoints'):
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
    
    # Get checkpoint with highest F1
    best_score, best_checkpoint = max(checkpoint_scores)
    print(f"Using checkpoint: {best_checkpoint} (F1: {best_score:.4f})")
    return str(best_checkpoint)

def evaluate_model(model_path, test_file):
    # Load model on CPU first
    model = EmotionClassifier.load_from_checkpoint(model_path, map_location='cpu')
    
    # Then move to MPS if available
    if torch.backends.mps.is_available():
        model = model.to('mps')
    
    model.eval()

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    # Load test data
    test_data = pd.read_csv(test_file)
    emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']

    # Predictions
    predictions = []
    true_labels = []

    with torch.no_grad():
        for _, row in test_data.iterrows():
            # Tokenize
            encoding = tokenizer(
                str(row['text']),
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Move to device
            input_ids = encoding['input_ids'].to('mps')
            attention_mask = encoding['attention_mask'].to('mps')

            # Get prediction
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu().numpy()
            predictions.append(preds[0])
            true_labels.append([row[label] for label in emotion_labels])

    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        true_labels, 
        predictions,
        target_names=emotion_labels
    ))

    # SHAP Analysis
    explainer = shap.Explainer(
        model,
        tokenizer,
        output_names=emotion_labels
    )
    
    # Get SHAP values for a subset of test data
    sample_texts = test_data['text'].iloc[:100].tolist()
    shap_values = explainer(sample_texts)

    # Save SHAP plots
    shap.summary_plot(
        shap_values, 
        sample_texts,
        class_names=emotion_labels,
        show=False
    )
    plt.savefig('shap_summary.png')
    plt.close()

def analyze_predictions(model, data_module, threshold=0.5):
    """Analyze model predictions with detailed metrics."""
    model.eval()
    device = model.device  # Get the model's current device
    all_preds = []
    all_labels = []
    
    # Get predictions
    val_loader = data_module.val_dataloader()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            logits = model(input_ids, attention_mask)
            preds = torch.sigmoid(logits)
            
            all_preds.append(preds.cpu())  # Move to CPU before appending
            all_labels.append(labels)
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Convert to binary predictions using threshold
    binary_preds = (all_preds > threshold).float()
    
    # Get metrics for each emotion
    emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
    
    print("\nDetailed Analysis:")
    print("-" * 50)
    
    # Plot confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, emotion in enumerate(emotion_labels):
        # Get confusion matrix
        cm = confusion_matrix(all_labels[:, idx], binary_preds[:, idx])
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{emotion.upper()}:")
        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Negatives: {tn}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
        axes[idx].set_title(f'{emotion.capitalize()} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('plots/confusion_matrices.png')
    plt.close()
    
    # Threshold analysis
    thresholds = np.arange(0.1, 1.0, 0.1)
    threshold_results = {emotion: {'precision': [], 'recall': [], 'f1': []} 
                        for emotion in emotion_labels}
    
    for thresh in thresholds:
        binary_preds = (all_preds > thresh).float()
        for idx, emotion in enumerate(emotion_labels):
            cm = confusion_matrix(all_labels[:, idx], binary_preds[:, idx])
            tn, fp, fn, tp = cm.ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            threshold_results[emotion]['precision'].append(precision)
            threshold_results[emotion]['recall'].append(recall)
            threshold_results[emotion]['f1'].append(f1)
    
    # Plot threshold analysis
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
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
    plt.savefig('plots/threshold_analysis.png')
    plt.close()
    
    return threshold_results

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
        test_file=f'{args.data_dir}/test/eng.csv'
    )
    data_module.setup()

    # Load model on CPU first
    model = EmotionClassifier.load_from_checkpoint(checkpoint_path, map_location='cpu')
    
    # Then move to MPS if available
    if torch.backends.mps.is_available():
        model = model.to('mps')
    
    results = analyze_predictions(model, data_module)

if __name__ == "__main__":
    main()
