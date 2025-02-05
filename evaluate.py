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

def evaluate(args):
    """Evaluate the model on test data and generate detailed metrics."""
    logging.basicConfig(level=logging.INFO)
    
    # Add Tensor Core optimization
    torch.set_float32_matmul_precision('high')
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Load model and move to GPU
    model = EmotionClassifier.load_from_checkpoint(args.checkpoint_path)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Initialize data module
    data_module = EmotionDataModule(
        data_dir=args.data_dir,
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
    
    print("\nGenerating predictions...")
    with torch.no_grad():
        for batch in data_module.val_dataloader():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            logits = model(batch['input_ids'], batch['attention_mask'])
            preds = torch.sigmoid(logits) > 0.5
            all_predictions.append(preds.cpu())
            all_labels.append(batch['labels'].cpu())
    
    predictions = torch.cat(all_predictions).numpy()
    true_labels = torch.cat(all_labels).numpy()
    
    # Generate metrics and save results
    emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
    
    print("\nDetailed Classification Report:")
    report = classification_report(
        true_labels, 
        predictions, 
        target_names=emotion_labels,
        zero_division=0
    )
    print(report)
    
    # Save outputs
    if args.plot_confusion:
        confusion_matrix_path = results_dir / 'confusion_matrix.png'
        plot_confusion_matrix(true_labels, predictions, emotion_labels, confusion_matrix_path)
    
    if args.output_predictions:
        predictions_path = results_dir / args.output_file
        save_predictions(predictions, true_labels, emotion_labels, predictions_path)
    
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

def main():
    parser = argparse.ArgumentParser()
    
    # Data and model arguments
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--checkpoint_path', type=str, required=True)
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