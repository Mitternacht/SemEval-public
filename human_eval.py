import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from pathlib import Path
import pandas as pd
import numpy as np
from model import EmotionClassifier
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import List, Dict
from transformers import RobertaTokenizer
from collections import Counter

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
    output_dir: str
) -> Dict:
    """
    Compare model predictions with human annotations on the same texts.
    
    Args:
        model_predictions: Model's predictions on test data
        human_annotations: DataFrame with human annotations
        emotion_labels: List of emotion labels
        output_dir: Directory to save comparison results
        
    Returns:
        Dictionary containing comparison metrics
    """
    # Ensure we're comparing the same texts
    text_ids = human_annotations['text_id'].values
    model_preds_subset = model_predictions[text_ids]
    
    # Convert human annotations to array format
    human_labels = human_annotations[emotion_labels].values
    
    # Calculate agreement metrics
    agreement_scores = []
    for i, emotion in enumerate(emotion_labels):
        # Calculate agreement for this emotion
        agreement = (model_preds_subset[:, i] == human_labels[:, i]).mean()
        agreement_scores.append({
            'emotion': emotion,
            'agreement': agreement
        })
    
    # Create agreement visualization
    plt.figure(figsize=(10, 6))
    agreement_df = pd.DataFrame(agreement_scores)
    sns.barplot(data=agreement_df, x='emotion', y='agreement')
    plt.title('Model-Human Agreement by Emotion')
    plt.ylabel('Agreement Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_human_agreement.png')
    plt.close()
    
    # Analyze disagreements
    disagreements = []
    for i, (pred, human) in enumerate(zip(model_preds_subset, human_labels)):
        if not np.array_equal(pred, human):
            disagreements.append({
                'text': human_annotations.iloc[i]['text'],
                'model_emotions': [e for j, e in enumerate(emotion_labels) if pred[j]],
                'human_emotions': [e for j, e in enumerate(emotion_labels) if human[j]]
            })
    
    # Calculate overall agreement
    overall_agreement = np.mean([score['agreement'] for score in agreement_scores])
    
    # Save detailed analysis
    with open(f'{output_dir}/human_comparison_analysis.txt', 'w') as f:
        f.write("Model-Human Comparison Analysis\n")
        f.write("==============================\n\n")
        
        # Overall statistics
        f.write(f"Total samples analyzed: {len(model_preds_subset)}\n")
        f.write(f"Overall agreement rate: {overall_agreement:.2%}\n")
        f.write(f"Total disagreements: {len(disagreements)}\n\n")
        
        # Per-emotion agreement
        f.write("Agreement by Emotion:\n")
        for score in agreement_scores:
            f.write(f"{score['emotion']}: {score['agreement']:.2%}\n")
        
        # Example disagreements
        f.write("\nExample Disagreements:\n")
        for i, d in enumerate(disagreements[:10]):  # Show first 10 disagreements
            f.write(f"\n{i+1}. Text: {d['text']}\n")
            f.write(f"   Model: {', '.join(d['model_emotions'])}\n")
            f.write(f"   Human: {', '.join(d['human_emotions'])}\n")
    
    return {
        'overall_agreement': overall_agreement,
        'emotion_agreements': {s['emotion']: s['agreement'] for s in agreement_scores},
        'num_disagreements': len(disagreements)
    }

def analyze_disagreements(
    model_predictions: np.ndarray,
    human_labels: np.ndarray,
    texts: List[str],
    emotion_labels: List[str],
    output_dir: str
) -> Dict:
    """
    Perform detailed analysis of model-human disagreements.
    
    Args:
        model_predictions: Model's binary predictions
        human_labels: Human annotator's binary labels
        texts: List of text samples
        emotion_labels: List of emotion labels
        output_dir: Directory to save analysis
    """
    disagreement_patterns = {
        'false_positives': {e: [] for e in emotion_labels},  # Model predicted yes, human no
        'false_negatives': {e: [] for e in emotion_labels},  # Model predicted no, human yes
        'common_contexts': {e: {'fp': [], 'fn': []} for e in emotion_labels},
        'co_occurrence': np.zeros((len(emotion_labels), len(emotion_labels)))
    }
    
    # Analyze each prediction
    for idx, (pred, human, text) in enumerate(zip(model_predictions, human_labels, texts)):
        for i, emotion in enumerate(emotion_labels):
            if pred[i] != human[i]:
                # Store false positives and negatives with context
                if pred[i] == 1:  # False positive
                    disagreement_patterns['false_positives'][emotion].append({
                        'text': text,
                        'other_emotions': [e for j, e in enumerate(emotion_labels) if human[j] == 1]
                    })
                else:  # False negative
                    disagreement_patterns['false_negatives'][emotion].append({
                        'text': text,
                        'other_emotions': [e for j, e in enumerate(emotion_labels) if human[j] == 1]
                    })
                    
                # Track emotion co-occurrences in disagreements
                for j, other_emotion in enumerate(emotion_labels):
                    if human[j] == 1:
                        disagreement_patterns['co_occurrence'][i, j] += 1
    
    # Generate insights report
    with open(f'{output_dir}/disagreement_analysis.txt', 'w') as f:
        f.write("Detailed Disagreement Analysis\n")
        f.write("============================\n\n")
        
        for emotion in emotion_labels:
            f.write(f"\nAnalysis for {emotion.upper()}:\n")
            f.write("------------------------\n")
            
            # False Positives Analysis
            fps = disagreement_patterns['false_positives'][emotion]
            f.write(f"\nFalse Positives (Model incorrectly predicted {emotion}):\n")
            f.write(f"Total cases: {len(fps)}\n")
            if fps:
                f.write("Common patterns:\n")
                # Analyze common words or phrases in false positives
                common_words = analyze_common_patterns([fp['text'] for fp in fps])
                for word, count in common_words.most_common(5):
                    f.write(f"- '{word}' appears in {count} cases\n")
                
                # Analyze co-occurring emotions in false positives
                co_emotions = [e for fp in fps for e in fp['other_emotions']]
                if co_emotions:
                    f.write("Common co-occurring emotions:\n")
                    for e, count in Counter(co_emotions).most_common():
                        f.write(f"- {e}: {count} cases\n")
            
            # False Negatives Analysis
            fns = disagreement_patterns['false_negatives'][emotion]
            f.write(f"\nFalse Negatives (Model missed {emotion}):\n")
            f.write(f"Total cases: {len(fns)}\n")
            if fns:
                f.write("Common patterns:\n")
                common_words = analyze_common_patterns([fn['text'] for fn in fns])
                for word, count in common_words.most_common(5):
                    f.write(f"- '{word}' appears in {count} cases\n")
                
                # Analyze co-occurring emotions in false negatives
                co_emotions = [e for fn in fns for e in fn['other_emotions']]
                if co_emotions:
                    f.write("Common co-occurring emotions:\n")
                    for e, count in Counter(co_emotions).most_common():
                        f.write(f"- {e}: {count} cases\n")
            
            # Example cases
            f.write("\nExample Cases:\n")
            for case_type, cases in [('False Positives', fps[:3]), ('False Negatives', fns[:3])]:
                f.write(f"\n{case_type}:\n")
                for case in cases:
                    f.write(f"Text: {case['text']}\n")
                    f.write(f"Other emotions present: {', '.join(case['other_emotions']) or 'none'}\n")
            
            # Improvement suggestions
            f.write("\nSuggested Improvements:\n")
            suggest_improvements(emotion, fps, fns, f)
    
    # Visualize co-occurrence patterns
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        disagreement_patterns['co_occurrence'],
        xticklabels=emotion_labels,
        yticklabels=emotion_labels,
        annot=True,
        fmt='d'
    )
    plt.title('Emotion Co-occurrence in Disagreements')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/disagreement_patterns.png')
    plt.close()
    
    return disagreement_patterns

def analyze_common_patterns(texts: List[str]) -> Counter:
    """Analyze common words/phrases in texts."""
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from collections import Counter
    
    # Get all words, excluding stopwords
    stop_words = set(stopwords.words('english'))
    words = [
        word.lower() for text in texts 
        for word in word_tokenize(text)
        if word.lower() not in stop_words and word.isalnum()
    ]
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
                annotator_dir
            )
            
            # Print results
            print(f"\nResults for Annotator {annotator_id}:")
            print(f"Overall agreement with human: {metrics['overall_agreement']:.2%}")
            print("\nAgreement by emotion:")
            for emotion, score in metrics['emotion_agreements'].items():
                print(f"{emotion}: {score:.2%}")

if __name__ == "__main__":
    main() 