import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pytorch_lightning as pl
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from model import EmotionClassifier
from data import EmotionDataModule
import logging
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import List, Dict
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter

def debug_predictions(true_labels: np.ndarray, predictions: np.ndarray, 
                     emotion_labels: List[str], num_samples: int = 5):
    """Debug helper to print detailed prediction information."""
    print("\nDEBUG: Prediction Analysis")
    print("========================")
    print(f"Shape of true_labels: {true_labels.shape}")
    print(f"Shape of predictions: {predictions.shape}")
    
    # Check for any NaN or invalid values
    print(f"\nNaN in true_labels: {np.isnan(true_labels).any()}")
    print(f"NaN in predictions: {np.isnan(predictions).any()}")
    
    # Print sample-wise comparison
    print(f"\nFirst {num_samples} samples comparison:")
    for i in range(min(num_samples, len(true_labels))):
        true_set = set(np.where(true_labels[i] == 1)[0])
        pred_set = set(np.where(predictions[i] == 1)[0])
        
        true_emotions = {emotion_labels[idx] for idx in true_set}
        pred_emotions = {emotion_labels[idx] for idx in pred_set}
        
        print(f"\nSample {i+1}:")
        print(f"True labels: {true_emotions}")
        print(f"Predictions: {pred_emotions}")
        print(f"Raw true: {true_labels[i]}")
        print(f"Raw pred: {predictions[i]}")
        
        # Calculate metrics for this sample
        sample_precision = len(true_set & pred_set) / len(pred_set) if pred_set else 1.0
        sample_recall = len(true_set & pred_set) / len(true_set) if true_set else 1.0
        sample_f1 = 2 * sample_precision * sample_recall / (sample_precision + sample_recall) if (sample_precision + sample_recall) > 0 else 0.0
        
        print(f"Sample metrics - Precision: {sample_precision:.4f}, Recall: {sample_recall:.4f}, F1: {sample_f1:.4f}")

def compute_multilabel_metrics(true_labels: np.ndarray, predictions: np.ndarray, emotion_labels: List[str]) -> Dict:
    """
    Compute comprehensive metrics for multi-label classification.
    
    Args:
        true_labels: Ground truth labels array
        predictions: Model predictions array
        emotion_labels: List of emotion names
        
    Returns:
        Dictionary containing metrics
    """
    metrics = {}
    
    # Per-emotion metrics (these are correct as they're calculated per emotion)
    per_emotion_metrics = {}
    for i, emotion in enumerate(emotion_labels):
        true_emotion = true_labels[:, i]
        pred_emotion = predictions[:, i]
        
        # Calculate metrics for this emotion
        per_emotion_metrics[emotion] = {
            'precision': precision_score(true_emotion, pred_emotion, zero_division=0),
            'recall': recall_score(true_emotion, pred_emotion, zero_division=0),
            'f1': f1_score(true_emotion, pred_emotion, zero_division=0),
            'support': np.sum(true_emotion)
        }
    
    # Compute sample-based metrics correctly
    sample_metrics = {
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for true, pred in zip(true_labels, predictions):
        true_set = set(np.where(true == 1)[0])
        pred_set = set(np.where(pred == 1)[0])
        
        # Calculate metrics for this sample
        if len(pred_set) > 0:
            precision = len(true_set & pred_set) / len(pred_set)
        else:
            precision = 1.0 if len(true_set) == 0 else 0.0
            
        if len(true_set) > 0:
            recall = len(true_set & pred_set) / len(true_set)
        else:
            recall = 1.0 if len(pred_set) == 0 else 0.0
        
        sample_metrics['precision'].append(precision)
        sample_metrics['recall'].append(recall)
        
        # Calculate F1 for this sample
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        sample_metrics['f1'].append(f1)
    
    # Calculate average sample-based metrics
    metrics['sample_metrics'] = {
        'precision': np.mean(sample_metrics['precision']),
        'recall': np.mean(sample_metrics['recall']),
        'f1': np.mean(sample_metrics['f1'])
    }
    
    # Calculate macro averages across emotions
    metrics['macro_metrics'] = {
        'precision': np.mean([m['precision'] for m in per_emotion_metrics.values()]),
        'recall': np.mean([m['recall'] for m in per_emotion_metrics.values()]),
        'f1': np.mean([m['f1'] for m in per_emotion_metrics.values()])
    }
    
    # Store per-emotion metrics
    metrics['per_emotion'] = per_emotion_metrics
    
    # Compute Jaccard similarity score per sample
    jaccard_scores = []
    for true, pred in zip(true_labels, predictions):
        true_set = set(np.where(true == 1)[0])
        pred_set = set(np.where(pred == 1)[0])
        if len(true_set | pred_set) == 0:
            jaccard_scores.append(1.0)
        else:
            jaccard_scores.append(len(true_set & pred_set) / len(true_set | pred_set))
    
    metrics['jaccard_similarity'] = np.mean(jaccard_scores)
    
    return metrics

def print_metrics_report(metrics: Dict, emotion_labels: List[str], file=None):
    """Print formatted metrics report."""
    def _print(text):
        print(text)
        if file:
            file.write(text + '\n')
    
    _print("\nPER-EMOTION METRICS")
    _print("==================")
    for emotion in emotion_labels:
        _print(f"\n{emotion.upper()}:")
        for metric, value in metrics['per_emotion'][emotion].items():
            if metric != 'support':
                _print(f"  {metric}: {value:.4f}")
            else:
                _print(f"  {metric}: {value}")
    
    _print("\nSAMPLE-BASED METRICS")
    _print("===================")
    for metric, value in metrics['sample_metrics'].items():
        _print(f"{metric}: {value:.4f}")
    
    _print("\nMACRO AVERAGES")
    _print("==============")
    for metric, value in metrics['macro_metrics'].items():
        _print(f"{metric}: {value:.4f}")
    
    _print(f"\nJaccard Similarity: {metrics['jaccard_similarity']:.4f}")

def analyze_prediction_discrepancies(true_labels: np.ndarray, predictions: np.ndarray, 
                                   texts: np.ndarray, emotion_labels: List[str]) -> Dict:
    """
    Analyze cases where per-label metrics and sample-based metrics differ.
    
    Args:
        true_labels: Ground truth labels
        predictions: Model predictions
        texts: Input texts
        emotion_labels: List of emotion names
        
    Returns:
        Dictionary containing analysis results
    """
    # Find samples where predictions don't exactly match ground truth
    mismatched_indices = []
    for i, (true, pred) in enumerate(zip(true_labels, predictions)):
        true_emotions = set(np.where(true == 1)[0])
        pred_emotions = set(np.where(pred == 1)[0])
        if true_emotions != pred_emotions:
            mismatched_indices.append(i)
    
    analysis = {
        'total_samples': len(true_labels),
        'mismatched_samples': len(mismatched_indices),
        'details': [],
        'common_patterns': {
            'missing_emotions': Counter(),
            'extra_emotions': Counter(),
            'confusion_pairs': Counter()  # Track common confusion patterns
        }
    }
    
    # Analyze each mismatch in detail
    for idx in mismatched_indices:
        true_emotions = set(np.where(true_labels[idx] == 1)[0])
        pred_emotions = set(np.where(predictions[idx] == 1)[0])
        
        # Convert indices to emotion names
        true_emotion_names = {emotion_labels[i] for i in true_emotions}
        pred_emotion_names = {emotion_labels[i] for i in pred_emotions}
        
        # Find missing and extra predictions
        missing = true_emotion_names - pred_emotion_names
        extra = pred_emotion_names - true_emotion_names
        
        # Update counters for common patterns
        for emotion in missing:
            analysis['common_patterns']['missing_emotions'][emotion] += 1
        for emotion in extra:
            analysis['common_patterns']['extra_emotions'][emotion] += 1
        
        # Track confusion pairs (what was predicted instead of what)
        for m in missing:
            for e in extra:
                analysis['common_patterns']['confusion_pairs'][(m, e)] += 1
        
        analysis['details'].append({
            'text': texts[idx],
            'true_emotions': true_emotion_names,
            'predicted_emotions': pred_emotion_names,
            'missing_emotions': missing,
            'extra_emotions': extra
        })
    
    return analysis

def debug_multilabel_metrics(true_labels: np.ndarray, predictions: np.ndarray, 
                           texts: np.ndarray, emotion_labels: List[str]) -> Dict:
    """
    Debug multi-label metrics calculation with detailed analysis.
    
    Args:
        true_labels: Ground truth labels array
        predictions: Model predictions array
        texts: Input text array
        emotion_labels: List of emotion names
        
    Returns:
        Dictionary containing analysis results
    """
    n_samples = len(true_labels)
    n_total_emotions = np.sum(true_labels)  # Total number of emotion labels
    
    print(f"\nDetailed Multi-label Analysis:")
    print(f"Number of samples (texts): {n_samples}")
    print(f"Total emotion labels: {n_total_emotions}")
    print(f"Average emotions per text: {n_total_emotions/n_samples:.2f}")
    
    # Analyze label distribution
    print("\nEmotion distribution per text:")
    emotions_per_text = np.sum(true_labels, axis=1)
    for i in range(1, int(max(emotions_per_text)) + 1):
        count = np.sum(emotions_per_text == i)
        percent = count/n_samples * 100
        print(f"Texts with {i} emotion(s): {count} ({percent:.1f}%)")
    
    # Compare predictions vs truth
    perfect_matches = np.all(predictions == true_labels, axis=1)
    n_perfect = np.sum(perfect_matches)
    print(f"\nTexts with perfect emotion matches: {n_perfect}/{n_samples} ({n_perfect/n_samples*100:.2f}%)")
    
    # Analyze per-emotion accuracy
    print("\nPer-emotion accuracy:")
    for i, emotion in enumerate(emotion_labels):
        correct = np.sum(predictions[:, i] == true_labels[:, i])
        accuracy = correct / n_samples
        print(f"{emotion}: {correct}/{n_samples} ({accuracy*100:.2f}%)")
    
    # Analyze multi-label cases specifically
    multi_label_indices = np.where(emotions_per_text > 1)[0]
    n_multi = len(multi_label_indices)
    
    if n_multi > 0:
        print(f"\nAnalysis of {n_multi} multi-label cases:")
        multi_perfect = np.sum(perfect_matches[multi_label_indices])
        print(f"Perfect predictions on multi-label: {multi_perfect}/{n_multi} ({multi_perfect/n_multi*100:.2f}%)")
        
        # Show examples of multi-label cases
        print("\nExample multi-label cases:")
        for idx in multi_label_indices[:5]:  # Show first 5 multi-label examples
            true_emotions = [emotion_labels[j] for j in range(len(emotion_labels)) if true_labels[idx][j]]
            pred_emotions = [emotion_labels[j] for j in range(len(emotion_labels)) if predictions[idx][j]]
            
            print(f"\nText: {texts[idx]}")
            print(f"True emotions: {true_emotions}")
            print(f"Predicted emotions: {pred_emotions}")
            print(f"Perfect match: {np.array_equal(true_labels[idx], predictions[idx])}")
    
    # Calculate detailed metrics for multi-label cases
    metrics = {
        'n_samples': n_samples,
        'n_total_emotions': n_total_emotions,
        'n_perfect_matches': n_perfect,
        'n_multi_label': n_multi,
        'emotion_distribution': {
            i: np.sum(emotions_per_text == i) 
            for i in range(1, int(max(emotions_per_text)) + 1)
        },
        'per_emotion_accuracy': {
            emotion: np.sum(predictions[:, i] == true_labels[:, i]) / n_samples
            for i, emotion in enumerate(emotion_labels)
        }
    }
    
    return metrics

def evaluate(args):
    """Evaluate the model on test data and generate detailed metrics."""
    logging.basicConfig(level=logging.INFO)
    
    # Add Tensor Core optimization
    torch.set_float32_matmul_precision('high')
    
    # Create results directory with timestamp
    results_dir = Path('results/evaluation')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nResults will be saved to: {results_dir}")
    
    # Load model and move to GPU
    model = EmotionClassifier.load_from_checkpoint(args.checkpoint_path)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Initialize data module with direct file path
    data_module = EmotionDataModule(
        data_path=args.data_path,  # Direct path to evaluation CSV
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    data_module.setup()
    
    # Run validation only once
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False  # Disable model summary for faster execution
    )
    
    results = trainer.validate(model, data_module.val_dataloader())
    
    # Get predictions in a single pass
    all_predictions = []
    all_labels = []
    all_probs = []  # Also store probabilities for threshold analysis
    
    print("\nGenerating predictions...")
    with torch.no_grad():
        for batch in data_module.val_dataloader():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            logits = model(batch['input_ids'], batch['attention_mask'])
            probs = torch.sigmoid(logits)
            preds = probs > 0.5
            all_predictions.append(preds.cpu())
            all_labels.append(batch['labels'].cpu())
            all_probs.append(probs.cpu())
    
    predictions = torch.cat(all_predictions).numpy()
    true_labels = torch.cat(all_labels).numpy()
    probabilities = torch.cat(all_probs).numpy()
    
    # After getting predictions and before computing standard metrics
    debug_metrics = debug_multilabel_metrics(
        true_labels,
        predictions,
        data_module.val_dataset.data['text'].values,
        ['anger', 'fear', 'joy', 'sadness', 'surprise']
    )
    
    # Generate and print metrics
    emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
    metrics = compute_multilabel_metrics(true_labels, predictions, emotion_labels)
    
    # Print metrics to console and file
    metrics_path = results_dir / 'evaluation_metrics.txt'
    with open(metrics_path, 'w') as f:
        print_metrics_report(metrics, emotion_labels, file=f)
    
    # After computing metrics, add discrepancy analysis
    print("\nAnalyzing prediction discrepancies...")
    discrepancy_analysis = analyze_prediction_discrepancies(
        true_labels, 
        predictions, 
        data_module.val_dataset.data['text'].values,
        emotion_labels
    )
    
    # Print analysis results
    print("\nDiscrepancy Analysis:")
    print(f"Total samples: {discrepancy_analysis['total_samples']}")
    print(f"Samples with mismatches: {discrepancy_analysis['mismatched_samples']} "
          f"({discrepancy_analysis['mismatched_samples']/discrepancy_analysis['total_samples']:.1%})")
    
    print("\nMost common missing emotions:")
    for emotion, count in discrepancy_analysis['common_patterns']['missing_emotions'].most_common(5):
        print(f"  {emotion}: {count} times")
    
    print("\nMost common extra emotions:")
    for emotion, count in discrepancy_analysis['common_patterns']['extra_emotions'].most_common(5):
        print(f"  {emotion}: {count} times")
    
    print("\nMost common confusion pairs (true → predicted):")
    for (true_emotion, pred_emotion), count in discrepancy_analysis['common_patterns']['confusion_pairs'].most_common(5):
        print(f"  {true_emotion} → {pred_emotion}: {count} times")
    
    print("\nDetailed examples of mismatches:")
    for i, detail in enumerate(discrepancy_analysis['details'][:5]):
        print(f"\nExample {i+1}:")
        print(f"Text: {detail['text']}")
        print(f"True emotions: {detail['true_emotions']}")
        print(f"Predicted emotions: {detail['predicted_emotions']}")
        print(f"Missing emotions: {detail['missing_emotions']}")
        print(f"Extra emotions: {detail['extra_emotions']}")
    
    # Add discrepancy analysis to metrics file
    with open(results_dir / 'evaluation_metrics.txt', 'a') as f:
        f.write("\n\nDISCREPANCY ANALYSIS\n")
        f.write("===================\n\n")
        f.write(f"Total samples: {discrepancy_analysis['total_samples']}\n")
        f.write(f"Samples with mismatches: {discrepancy_analysis['mismatched_samples']} "
                f"({discrepancy_analysis['mismatched_samples']/discrepancy_analysis['total_samples']:.1%})\n\n")
        
        f.write("Most common missing emotions:\n")
        for emotion, count in discrepancy_analysis['common_patterns']['missing_emotions'].most_common():
            f.write(f"  {emotion}: {count} times\n")
        
        f.write("\nMost common extra emotions:\n")
        for emotion, count in discrepancy_analysis['common_patterns']['extra_emotions'].most_common():
            f.write(f"  {emotion}: {count} times\n")
        
        f.write("\nMost common confusion pairs (true → predicted):\n")
        for (true_emotion, pred_emotion), count in discrepancy_analysis['common_patterns']['confusion_pairs'].most_common():
            f.write(f"  {true_emotion} → {pred_emotion}: {count} times\n")
        
        f.write("\nDetailed examples of mismatches:\n")
        for i, detail in enumerate(discrepancy_analysis['details'][:10]):
            f.write(f"\nExample {i+1}:\n")
            f.write(f"Text: {detail['text']}\n")
            f.write(f"True emotions: {detail['true_emotions']}\n")
            f.write(f"Predicted emotions: {detail['predicted_emotions']}\n")
            f.write(f"Missing emotions: {detail['missing_emotions']}\n")
            f.write(f"Extra emotions: {detail['extra_emotions']}\n")
    
    # Add debug metrics to the results file
    with open(results_dir / 'evaluation_metrics.txt', 'a') as f:
        f.write("\n\nDETAILED MULTI-LABEL ANALYSIS\n")
        f.write("===========================\n\n")
        f.write(f"Total samples: {debug_metrics['n_samples']}\n")
        f.write(f"Total emotion labels: {debug_metrics['n_total_emotions']}\n")
        f.write(f"Average emotions per text: {debug_metrics['n_total_emotions']/debug_metrics['n_samples']:.2f}\n\n")
        
        f.write("Emotion distribution per text:\n")
        for n_emotions, count in debug_metrics['emotion_distribution'].items():
            percent = count/debug_metrics['n_samples'] * 100
            f.write(f"Texts with {n_emotions} emotion(s): {count} ({percent:.1f}%)\n")
        
        f.write(f"\nPerfect matches: {debug_metrics['n_perfect_matches']}/{debug_metrics['n_samples']} ")
        f.write(f"({debug_metrics['n_perfect_matches']/debug_metrics['n_samples']*100:.2f}%)\n\n")
        
        f.write("Per-emotion accuracy:\n")
        for emotion, accuracy in debug_metrics['per_emotion_accuracy'].items():
            f.write(f"{emotion}: {accuracy*100:.2f}%\n")
    
    # Save predictions with more details
    if args.output_predictions:
        save_detailed_predictions(
            predictions,
            probabilities,  # Now also including probabilities
            true_labels,
            data_module.val_dataset.data['text'].values,
            emotion_labels,
            results_dir / f"detailed_{args.output_file}"
        )
        print(f"\nDetailed predictions saved to: {results_dir / f'detailed_{args.output_file}'}")
    
    # Save confusion matrices
    if args.plot_confusion:
        plot_confusion_matrix(true_labels, predictions, emotion_labels, results_dir / 'confusion_matrix.png')
        print(f"Confusion matrices saved to: {results_dir / 'confusion_matrix.png'}")
    
    print(f"\nEvaluation metrics saved to: {results_dir / 'evaluation_metrics.txt'}")
    
    return results

def plot_confusion_matrix(true_labels, predictions, emotion_labels, save_path):
    """Plot confusion matrix for each emotion."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, emotion in enumerate(emotion_labels):
        cm = confusion_matrix(true_labels[:, idx], predictions[:, idx])
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            ax=axes[idx],
            xticklabels=['Not ' + emotion, emotion],
            yticklabels=['Not ' + emotion, emotion]
        )
        axes[idx].set_title(f'Confusion Matrix - {emotion.capitalize()}')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nConfusion matrix saved to {save_path}")

def save_predictions(predictions, true_labels, emotion_labels, output_file):
    """Save predictions and true labels to CSV."""
    results_df = pd.DataFrame({
        **{f'pred_{emotion}': predictions[:, i] for i, emotion in enumerate(emotion_labels)},
        **{f'true_{emotion}': true_labels[:, i] for i, emotion in enumerate(emotion_labels)}
    })
    results_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")

def save_detailed_predictions(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    texts: np.ndarray,
    emotion_labels: List[str],
    output_file: Path
):
    """Save detailed predictions including probabilities and text."""
    results_df = pd.DataFrame({
        'text': texts,
        **{f'pred_{emotion}': predictions[:, i] for i, emotion in enumerate(emotion_labels)},
        **{f'prob_{emotion}': probabilities[:, i] for i, emotion in enumerate(emotion_labels)},
        **{f'true_{emotion}': true_labels[:, i] for i, emotion in enumerate(emotion_labels)}
    })
    
    # Add summary columns
    results_df['predicted_emotions'] = results_df[[f'pred_{e}' for e in emotion_labels]].apply(
        lambda x: ', '.join([e for e, p in zip(emotion_labels, x) if p]), axis=1
    )
    results_df['true_emotions'] = results_df[[f'true_{e}' for e in emotion_labels]].apply(
        lambda x: ', '.join([e for e, p in zip(emotion_labels, x) if p]), axis=1
    )
    results_df['correct'] = results_df['predicted_emotions'] == results_df['true_emotions']
    
    # Calculate Jaccard similarity per sample
    results_df['jaccard_similarity'] = results_df.apply(
        lambda row: len(set(row['predicted_emotions'].split(', ')) & set(row['true_emotions'].split(', '))) / 
                   len(set(row['predicted_emotions'].split(', ')) | set(row['true_emotions'].split(', ')))
        if row['predicted_emotions'] or row['true_emotions'] else 1.0,
        axis=1
    )
    
    results_df.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser()
    
    # Changed data_dir to data_path
    parser.add_argument('--data_path', type=str, required=True, help='Path to evaluation CSV file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Output arguments
    parser.add_argument('--output_predictions', action='store_true')
    parser.add_argument('--output_file', type=str, default='predictions.csv')
    parser.add_argument('--plot_confusion', action='store_true')
    
    args = parser.parse_args()
    
    evaluate(args)

if __name__ == "__main__":
    main()