import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from pathlib import Path
import pandas as pd
import numpy as np
from model import EmotionClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import List, Dict, Tuple
from transformers import RobertaTokenizer
from collections import Counter
import shap
import logging

def get_model_predictions(model: EmotionClassifier, test_data: pd.DataFrame) -> np.ndarray:
    """
    Generate model predictions for unlabeled test data.
    
    Args:
        model: Trained EmotionClassifier model
        test_data: DataFrame containing unlabeled test texts
        
    Returns:
        numpy array of predictions
    """
    model.eval()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Create test dataset
    encodings = tokenizer(
        test_data['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Get predictions
    all_predictions = []
    batch_size = 32
    
    print("\nGenerating predictions...")
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch_input_ids = encodings['input_ids'][i:i+batch_size].to(model.device)
            batch_attention_mask = encodings['attention_mask'][i:i+batch_size].to(model.device)
            
            logits = model(batch_input_ids, batch_attention_mask)
            preds = torch.sigmoid(logits) > 0.5
            all_predictions.append(preds.cpu())
    
    return torch.cat(all_predictions).numpy()

def collect_human_annotations(text_data: pd.DataFrame, output_file: str, annotator_id: str = None) -> None:
    """
    Tool to collect human annotations for the unlabeled test data.
    Creates a new annotation file for each annotator or resumes existing one.
    
    Args:
        text_data: DataFrame containing unlabeled texts
        output_file: Base path for annotation files (will append annotator_id)
        annotator_id: Optional identifier for the annotator
    """
    emotions = ['anger', 'fear', 'joy', 'sadness', 'surprise']
    
    # Get or create annotator ID
    if annotator_id is None:
        existing_files = list(Path(output_file).parent.glob('human_annotations_*.csv'))
        existing_ids = [f.stem.split('_')[-1] for f in existing_files]
        
        print("\nExisting annotation files:")
        if existing_ids:
            for id_ in existing_ids:
                print(f"- Annotator {id_}")
            choice = input("\nEnter existing ID to resume, or new ID for fresh annotations: ")
        else:
            print("No existing annotations found.")
            choice = input("Enter annotator ID (e.g., 'A1'): ")
        
        annotator_id = choice
    
    # Construct file path with annotator ID
    file_stem = Path(output_file).stem
    file_suffix = Path(output_file).suffix
    annotation_file = Path(output_file).parent / f"{file_stem}_{annotator_id}{file_suffix}"
    
    annotations = []
    start_idx = 0
    
    # Check if this annotator has existing work
    if annotation_file.exists():
        existing_annotations = pd.read_csv(annotation_file)
        annotations = existing_annotations.to_dict('records')
        start_idx = len(annotations)
        print(f"\nResuming annotations for Annotator {annotator_id} from text {start_idx + 1}")
    else:
        print(f"\nStarting new annotation set for Annotator {annotator_id}")
    
    print("\nHuman Annotation Collection")
    print("==========================")
    print("For each text, enter y/n for each emotion (or 'q' to quit and save)\n")
    
    try:
        for idx, row in text_data.iloc[start_idx:].iterrows():
            print(f"\nText [{idx+1}/{len(text_data)}]:")
            print(f"{row['text']}\n")
            
            annotation = {
                'text': row['text'],
                'text_id': idx  # Keep track of original text index
            }
            
            for emotion in emotions:
                while True:
                    response = input(f"Does this text express {emotion}? (y/n/q): ").lower()
                    if response == 'q':
                        # Save partial annotations and exit
                        pd.DataFrame(annotations).to_csv(annotation_file, index=False)
                        print(f"\nProgress saved to {annotation_file}")
                        return
                    if response in ['y', 'n']:
                        annotation[emotion] = 1 if response == 'y' else 0
                        break
                    print("Please enter 'y' or 'n'")
            
            annotations.append(annotation)
            
            # Save after each annotation
            pd.DataFrame(annotations).to_csv(annotation_file, index=False)
            print("\n---")
            
    except KeyboardInterrupt:
        print("\nAnnotation interrupted. Saving progress...")
        pd.DataFrame(annotations).to_csv(annotation_file, index=False)
        print(f"Progress saved to {annotation_file}")
        return

def compare_human_model_predictions(
    model_predictions: np.ndarray,
    human_annotations: pd.DataFrame,
    emotion_labels: List[str],
    output_dir: str,
    model: EmotionClassifier,
    tokenizer: RobertaTokenizer
) -> Dict:
    """Compare model predictions with human annotations."""
    # Ensure we're comparing the same texts
    text_ids = human_annotations['text_id'].values
    model_preds_subset = model_predictions[text_ids]
    
    # Convert human annotations to array format
    human_labels = human_annotations[emotion_labels].values
    
    # Add new metrics
    metrics = {
        'per_emotion': {},
        'overall': {}
    }
    
    for i, emotion in enumerate(emotion_labels):
        # Calculate detailed metrics per emotion
        tp = np.sum((model_preds_subset[:, i] == 1) & (human_labels[:, i] == 1))
        fp = np.sum((model_preds_subset[:, i] == 1) & (human_labels[:, i] == 0))
        fn = np.sum((model_preds_subset[:, i] == 0) & (human_labels[:, i] == 1))
        tn = np.sum((model_preds_subset[:, i] == 0) & (human_labels[:, i] == 0))
        
        # Calculate metrics
        metrics['per_emotion'][emotion] = {
            'agreement': (tp + tn) / (tp + tn + fp + fn),
            'disagreement_rate': (fp + fn) / (tp + tn + fp + fn),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        }
    
    # Calculate overall metrics
    metrics['overall'] = {
        'exact_match_ratio': np.mean(np.all(model_preds_subset == human_labels, axis=1)),
        'hamming_loss': np.mean(model_preds_subset != human_labels),
        'subset_accuracy': np.mean(np.all(model_preds_subset == human_labels, axis=1))
    }
    
    # Get disagreement analysis
    disagreements = analyze_disagreements(
        model_preds_subset,
        human_labels,
        human_annotations['text'].values,
        emotion_labels,
        output_dir,
        model,
        tokenizer
    )
    
    # Save raw comparisons
    save_raw_comparisons(
        model_preds_subset,
        human_labels,
        human_annotations['text'].values,
        emotion_labels,
        output_dir
    )
    
    # Print results
    print(f"\nResults for Annotator:")
    print(f"Overall agreement with human: {metrics['overall']['exact_match_ratio']:.2%}")
    print("\nAgreement by emotion:")
    for emotion, score in metrics['per_emotion'].items():
        print(f"{emotion}:")
        for metric, value in score.items():
            print(f"  {metric}: {value:.2%}")
    
    return metrics

def analyze_disagreements(
    model_predictions: np.ndarray,
    human_labels: np.ndarray,
    texts: List[str],
    emotion_labels: List[str],
    output_dir: str,
    model: EmotionClassifier,
    tokenizer: RobertaTokenizer
) -> Dict:
    """Perform detailed analysis of model-human disagreements."""
    
    # Calculate metrics first
    metrics = {
        'per_emotion': {},
        'overall': {
            'exact_match_ratio': np.mean(np.all(model_predictions == human_labels, axis=1)),
            'hamming_loss': np.mean(model_predictions != human_labels)
        }
    }
    
    for i, emotion in enumerate(emotion_labels):
        tp = np.sum((model_predictions[:, i] == 1) & (human_labels[:, i] == 1))
        fp = np.sum((model_predictions[:, i] == 1) & (human_labels[:, i] == 0))
        fn = np.sum((model_predictions[:, i] == 0) & (human_labels[:, i] == 1))
        tn = np.sum((model_predictions[:, i] == 0) & (human_labels[:, i] == 0))
        
        metrics['per_emotion'][emotion] = {
            'agreement': (tp + tn) / (tp + tn + fp + fn),
            'disagreement_rate': (fp + fn) / (tp + tn + fp + fn),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        }
    
    # Create HTML report with metrics and improved SHAP visualization
    html_report_path = Path(output_dir) / 'disagreement_analysis.html'
    with open(html_report_path, 'w') as f:
        # Write the static part of HTML first
        f.write("""
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .example { margin: 20px 0; padding: 10px; border: 1px solid #ccc; }
                .labels { margin: 10px 0; }
                .model-prediction { color: #2c5282; }
                .human-annotation { color: #744210; }
                .visualization { margin: 10px 0; }
                .metrics { background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }
                .metric-value { font-weight: bold; color: #2c5282; }
            </style>
        </head>
        <body>
        <h1>Disagreement Analysis with SHAP Visualization</h1>
        """)
        
        # Write the metrics section separately with proper formatting
        f.write(f"""
        <div class="metrics">
            <h2>Overall Metrics</h2>
            <p>Exact Match Ratio: <span class="metric-value">{metrics['overall']['exact_match_ratio']:.2%}</span></p>
            <p>Hamming Loss: <span class="metric-value">{metrics['overall']['hamming_loss']:.3f}</span></p>
        </div>
        """)
        
        for emotion_idx, emotion in enumerate(emotion_labels):
            f.write(f"<h2>Analysis for {emotion.upper()}</h2>")
            
            # Add emotion-specific metrics
            f.write('<div class="metrics">')
            f.write("<h3>Metrics</h3>")
            for metric, value in metrics['per_emotion'][emotion].items():
                f.write(f'<p>{metric.replace("_", " ").title()}: <span class="metric-value">{value:.2%}</span></p>')
            f.write('</div>')
            
            # False Positives Analysis
            fps = []
            for idx, (pred, human) in enumerate(zip(model_predictions[:, emotion_idx], human_labels[:, emotion_idx])):
                if pred == 1 and human == 0:
                    human_emotions = [
                        emotion_labels[i] 
                        for i in range(len(emotion_labels)) 
                        if human_labels[idx, i] == 1
                    ]
                    fps.append({
                        'text': texts[idx],
                        'human_emotions': human_emotions
                    })
            
            if fps:
                f.write("<h3>Model Over-predictions</h3>")
                for i, case in enumerate(fps[:3], 1):
                    f.write('<div class="example">')
                    f.write(f"<h4>Example {i}</h4>")
                    f.write('<div class="labels">')
                    f.write(f'<div class="model-prediction">MODEL: Predicted "{emotion}"</div>')
                    f.write('<div class="human-annotation">HUMAN: ')
                    if case['human_emotions']:
                        f.write(f'Annotated as: {", ".join(case["human_emotions"])}')
                    else:
                        f.write('Did not mark any emotions')
                    f.write('</div></div>')
                    
                    # Add improved SHAP visualization
                    try:
                        shap_values = get_shap_values(
                            case['text'],
                            model,
                            tokenizer,
                            emotion_idx
                        )
                        html_viz = create_shap_visualization(
                            case['text'],
                            shap_values,
                            tokenizer
                        )
                        f.write(f'<div class="visualization">{html_viz}</div>')
                    except Exception as e:
                        logging.error(f"SHAP visualization failed: {e}")
                        f.write('<div class="error">SHAP visualization failed</div>')
                    
                    f.write('</div>')
            
            # False Negatives Analysis
            fns = []
            for idx, (pred, human) in enumerate(zip(model_predictions[:, emotion_idx], human_labels[:, emotion_idx])):
                if pred == 0 and human == 1:
                    model_emotions = [
                        emotion_labels[i] 
                        for i in range(len(emotion_labels)) 
                        if model_predictions[idx, i] == 1
                    ]
                    fns.append({
                        'text': texts[idx],
                        'model_emotions': model_emotions
                    })
            
            if fns:
                f.write("<h3>Model Under-predictions</h3>")
                for i, case in enumerate(fns[:3], 1):
                    f.write('<div class="example">')
                    f.write(f"<h4>Example {i}</h4>")
                    f.write('<div class="labels">')
                    f.write(f'<div class="human-annotation">HUMAN: Annotated as "{emotion}"</div>')
                    f.write('<div class="model-prediction">MODEL: ')
                    if case['model_emotions']:
                        f.write(f'Predicted: {", ".join(case["model_emotions"])}')
                    else:
                        f.write('Did not predict any emotions')
                    f.write('</div></div>')
                    
                    # Add improved SHAP visualization
                    try:
                        shap_values = get_shap_values(
                            case['text'],
                            model,
                            tokenizer,
                            emotion_idx
                        )
                        html_viz = create_shap_visualization(
                            case['text'],
                            shap_values,
                            tokenizer
                        )
                        f.write(f'<div class="visualization">{html_viz}</div>')
                    except Exception as e:
                        logging.error(f"SHAP visualization failed: {e}")
                        f.write('<div class="error">SHAP visualization failed</div>')
                    
                    f.write('</div>')
        
        f.write("</body></html>")

    # Generate text report
    with open(f'{output_dir}/disagreement_analysis.txt', 'w') as f:
        f.write("DETAILED DISAGREEMENT ANALYSIS\n")
        f.write("============================\n\n")
        
        for emotion_idx, emotion in enumerate(emotion_labels):
            f.write(f"\nANALYSIS FOR {emotion.upper()}\n")
            f.write("=" * (len(emotion) + 20) + "\n")
            
            # False Positives Analysis
            fps = []
            for idx, (pred, human) in enumerate(zip(model_predictions[:, emotion_idx], human_labels[:, emotion_idx])):
                if pred == 1 and human == 0:
                    human_emotions = [
                        emotion_labels[i] 
                        for i in range(len(emotion_labels)) 
                        if human_labels[idx, i] == 1
                    ]
                    fps.append({
                        'text': texts[idx],
                        'human_emotions': human_emotions
                    })
            
            if fps:
                f.write(f"\nMODEL OVER-PREDICTION ({emotion}):\n")
                f.write(f"Cases where model predicted {emotion} but human did not: {len(fps)}\n")
                
                for i, case in enumerate(fps[:3], 1):
                    f.write(f"\nExample {i}:\n")
                    f.write(f"Text: {case['text']}\n")
                    f.write("MODEL: Predicted " + emotion + "\n")
                    if case['human_emotions']:
                        f.write("HUMAN: Annotated as: " + ", ".join(case['human_emotions']) + "\n")
                    else:
                        f.write("HUMAN: Did not mark any emotions\n")
            
            # False Negatives Analysis
            fns = []
            for idx, (pred, human) in enumerate(zip(model_predictions[:, emotion_idx], human_labels[:, emotion_idx])):
                if pred == 0 and human == 1:
                    model_emotions = [
                        emotion_labels[i] 
                        for i in range(len(emotion_labels)) 
                        if model_predictions[idx, i] == 1
                    ]
                    fns.append({
                        'text': texts[idx],
                        'model_emotions': model_emotions
                    })
            
            if fns:
                f.write(f"\nMODEL UNDER-PREDICTION ({emotion}):\n")
                f.write(f"Cases where human marked {emotion} but model missed it: {len(fns)}\n")
                
                for i, case in enumerate(fns[:3], 1):
                    f.write(f"\nExample {i}:\n")
                    f.write(f"Text: {case['text']}\n")
                    f.write("HUMAN: Annotated as " + emotion + "\n")
                    if case['model_emotions']:
                        f.write("MODEL: Predicted instead: " + ", ".join(case['model_emotions']) + "\n")
                    else:
                        f.write("MODEL: Did not predict any emotions\n")
            
            f.write("\n" + "="*50 + "\n")
    
    return {
        'false_positives': fps,
        'false_negatives': fns
    }

def analyze_common_patterns(texts: List[str]) -> Counter:
    """Analyze common words/phrases in texts."""
    from collections import Counter
    
    # Simple word splitting and basic cleaning
    words = []
    for text in texts:
        # Split on whitespace and convert to lowercase
        text_words = text.lower().split()
        # Only keep words with 3+ characters
        text_words = [w for w in text_words if len(w) >= 3]
        words.extend(text_words)
    
    return Counter(words)

def suggest_improvements(emotion: str, false_positives: List[Dict], 
                       false_negatives: List[Dict], file) -> None:
    """Generate improvement suggestions based on error patterns."""
    
    # Calculate error ratios
    total_errors = len(false_positives) + len(false_negatives)
    fp_ratio = len(false_positives) / total_errors if total_errors > 0 else 0
    
    suggestions = []
    
    # Analyze error distribution
    if fp_ratio > 0.7:
        suggestions.append(
            f"Model is too aggressive in predicting {emotion}. "
            "Consider increasing the classification threshold."
        )
    elif fp_ratio < 0.3:
        suggestions.append(
            f"Model is too conservative in predicting {emotion}. "
            "Consider decreasing the classification threshold."
        )
    
    # Analyze common patterns in errors
    fp_words = analyze_common_patterns([fp['text'] for fp in false_positives])
    fn_words = analyze_common_patterns([fn['text'] for fn in false_negatives])
    
    # Suggest based on common words
    if fp_words:
        suggestions.append(
            f"Common words in false positives: {', '.join(w for w, _ in fp_words.most_common(3))}. "
            "Consider adding examples with these words but different emotions."
        )
    if fn_words:
        suggestions.append(
            f"Common words in false negatives: {', '.join(w for w, _ in fn_words.most_common(3))}. "
            "Consider adding more training examples with these patterns."
        )
    
    # Write suggestions
    file.write("\nImprovement Suggestions:\n")
    for i, suggestion in enumerate(suggestions, 1):
        file.write(f"{i}. {suggestion}\n")

def get_shap_values(text: str, model: EmotionClassifier, tokenizer: RobertaTokenizer, emotion_idx: int):
    """Get SHAP values for a single text with improved implementation and memory management."""
    try:
        # Move model to CPU for SHAP analysis to avoid memory issues
        model = model.cpu()
        
        # Tokenize with proper handling
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128,
            return_special_tokens_mask=True
        )
        
        # Create smaller background data
        background_size = 10  # Reduced from 100
        background_data = []
        max_len = len(inputs['input_ids'][0])
        for _ in range(background_size):
            random_ids = torch.randint(0, tokenizer.vocab_size, (1, max_len))
            background_data.append(random_ids[0].numpy())
        background_data = np.array(background_data)
        
        # Create explainer function with proper masking
        def f(x):
            try:
                if isinstance(x, str):
                    x = [x]
                x = np.array(x)
                with torch.no_grad():
                    x_tensor = torch.tensor(x, dtype=torch.long)
                    attention_mask = (x_tensor != tokenizer.pad_token_id).long()
                    output = model(x_tensor, attention_mask)
                    probs = torch.sigmoid(output)[:, emotion_idx].numpy()
                    return probs
            except Exception as e:
                logging.error(f"Error in SHAP prediction function: {e}")
                return np.zeros(len(x))
        
        # Initialize explainer with reduced parameters
        explainer = shap.KernelExplainer(
            f, 
            background_data,
            link="identity"  # Changed from logit to avoid division by zero
        )
        
        # Calculate SHAP values with reduced sample size
        token_length = len(inputs['input_ids'][0])
        min_samples = 50  # Reduced from 100
        dynamic_samples = min(max(min_samples, token_length * 5), 100)  # Cap at 100
        
        shap_values = explainer.shap_values(
            inputs['input_ids'][0].numpy(),
            nsamples=dynamic_samples,
            l1_reg=False  # Disable regularization to avoid NaN issues
        )
        
        # Handle potential NaN values
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Take first element if list
        shap_values = np.nan_to_num(shap_values)  # Replace NaN with 0
        
        # Filter out special tokens
        special_tokens_mask = inputs['special_tokens_mask'][0].numpy()
        shap_values = shap_values * (1 - special_tokens_mask)
        
        return shap_values
        
    except Exception as e:
        logging.error(f"Error calculating SHAP values: {e}")
        return np.zeros(token_length)
    finally:
        # Move model back to original device
        if torch.cuda.is_available():
            model = model.cuda()

def create_shap_visualization(text: str, shap_values: np.ndarray, tokenizer: RobertaTokenizer):
    """Create improved SHAP visualization with better handling of special cases."""
    # Get tokens with proper handling of special tokens
    tokens = tokenizer.tokenize(text)
    special_tokens = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}
    
    # Filter out special tokens and their SHAP values
    token_shap = [
        (token, value) 
        for token, value in zip(tokens, shap_values)
        if token not in special_tokens
    ]
    
    # Normalize SHAP values to [0, 1] range for color intensity
    max_abs_shap = max(abs(value) for _, value in token_shap)
    
    html_parts = []
    for token, value in token_shap:
        # Clean the token - remove the Ġ prefix
        clean_token = token.replace('Ġ', '')
        
        # Normalize value to [0, 1]
        intensity = min(abs(value) / max_abs_shap, 1.0)
        
        # Choose color based on positive/negative contribution
        if value > 0:
            # Green for positive influence (pushes toward emotion)
            color = f"background-color: rgba(0, 255, 0, {intensity:.2f})"
        else:
            # Red for negative influence (pushes away from emotion)
            color = f"background-color: rgba(255, 0, 0, {intensity:.2f})"
        
        # Create span with tooltip showing SHAP value
        span = f'<span style="{color}" title="SHAP: {value:.3f}">{clean_token}</span>'
        html_parts.append(span)
    
    # Join tokens with spaces
    html = " ".join(html_parts)
    
    # Add legend
    legend = """
    <div style="margin-top: 10px; font-size: 0.9em;">
        <span style="background-color: rgba(0, 255, 0, 0.5)">Green</span>: positive influence
        <span style="margin-left: 10px; background-color: rgba(255, 0, 0, 0.5)">Red</span>: negative influence
        (Darker color = stronger influence)
    </div>
    """
    
    return f"<div style='font-family: monospace; white-space: pre-wrap;'>\n{html}\n{legend}\n</div>"

def save_raw_comparisons(
    model_predictions: np.ndarray,
    human_labels: np.ndarray,
    texts: List[str],
    emotion_labels: List[str],
    output_dir: str
):
    """Save raw comparison data between model and human annotations."""
    
    with open(f'{output_dir}/raw_comparisons.txt', 'w') as f:
        f.write("RAW MODEL VS HUMAN COMPARISONS\n")
        f.write("=============================\n\n")
        
        for idx, (text, model_preds, human_preds) in enumerate(zip(texts, model_predictions, human_labels), 1):
            f.write(f"Text {idx}: {text}\n")
            f.write("MODEL: ")
            model_emotions = [emotion_labels[i] for i, pred in enumerate(model_preds) if pred == 1]
            f.write(", ".join(model_emotions) if model_emotions else "none")
            f.write("\nHUMAN: ")
            human_emotions = [emotion_labels[i] for i, pred in enumerate(human_preds) if pred == 1]
            f.write(", ".join(human_emotions) if human_emotions else "none")
            f.write("\n\n")
            f.write("-" * 50 + "\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, default='data/semeval/test/eng.csv',
                      help='Path to unlabeled test data')
    parser.add_argument('--model_checkpoint', type=str,
                      help='Path to model checkpoint (only needed for comparison)')
    parser.add_argument('--collect_annotations', action='store_true',
                      help='Collect human annotations')
    parser.add_argument('--annotations_dir', type=str, default='annotations',
                      help='Directory to save/load human annotations')
    parser.add_argument('--annotator_id', type=str,
                      help='Annotator ID (e.g., A1, A2)')
    parser.add_argument('--compare_all', action='store_true',
                      help='Compare with all available annotators')
    parser.add_argument('--output_dir', type=str, default='results/human_eval',
                      help='Directory to save results')
    args = parser.parse_args()
    
    # Create directories
    Path(args.annotations_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Base filename for annotations
    annotations_file = Path(args.annotations_dir) / 'human_annotations.csv'
    
    # Load test data
    test_data = pd.read_csv(args.test_data)
    
    if args.collect_annotations:
        # Collect human annotations
        collect_human_annotations(test_data, annotations_file, args.annotator_id)
    else:
        # For comparison, we need the model checkpoint
        if not args.model_checkpoint:
            print("Error: Model checkpoint required for comparison")
            return
            
        # Load model and get predictions
        model = EmotionClassifier.load_from_checkpoint(args.model_checkpoint)
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get model predictions for all test data
        predictions = get_model_predictions(model, test_data)
        
        # Determine which annotations to compare
        if args.compare_all:
            # Compare with all annotation files
            annotation_files = list(Path(args.annotations_dir).glob('human_annotations_*.csv'))
        elif args.annotator_id:
            # Compare with specific annotator
            specific_file = Path(args.annotations_dir) / f'human_annotations_{args.annotator_id}.csv'
            if not specific_file.exists():
                print(f"No annotations found for Annotator {args.annotator_id}")
                return
            annotation_files = [specific_file]
        else:
            # List available annotators and ask user
            annotation_files = list(Path(args.annotations_dir).glob('human_annotations_*.csv'))
            if not annotation_files:
                print("No annotation files found!")
                return
                
            print("\nAvailable annotators:")
            for i, f in enumerate(annotation_files):
                annotator = f.stem.split('_')[-1]
                print(f"{i+1}. Annotator {annotator}")
            
            choice = input("\nEnter number(s) to compare (comma-separated), or 'all': ")
            if choice.lower() == 'all':
                pass  # Use all files
            else:
                try:
                    indices = [int(x.strip())-1 for x in choice.split(',')]
                    annotation_files = [annotation_files[i] for i in indices]
                except (ValueError, IndexError):
                    print("Invalid selection")
                    return
        
        # Compare with each annotator
        emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
        for ann_file in annotation_files:
            annotator_id = ann_file.stem.split('_')[-1]
            print(f"\nAnalyzing annotations from Annotator {annotator_id}...")
            
            # Load human annotations
            human_annotations = pd.read_csv(ann_file)
            
            # Create output directory for this annotator
            annotator_dir = Path(args.output_dir) / f"annotator_{annotator_id}"
            annotator_dir.mkdir(parents=True, exist_ok=True)
            
            # Compare predictions
            metrics = compare_human_model_predictions(
                predictions,
                human_annotations,
                emotion_labels,
                annotator_dir,
                model,
                RobertaTokenizer.from_pretrained('roberta-base')
            )
            
            # Print results
            print(f"\nResults for Annotator {annotator_id}:")
            print(f"Overall agreement with human: {metrics['overall']['exact_match_ratio']:.2%}")
            print("\nAgreement by emotion:")
            for emotion, score in metrics['per_emotion'].items():
                print(f"{emotion}:")
                for metric, value in score.items():
                    print(f"  {metric}: {value:.2%}")

if __name__ == "__main__":
    main() 