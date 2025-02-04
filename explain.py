import torch
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer
from model import EmotionClassifier
from typing import List, Dict, Any

class EmotionExplainer:
    def __init__(self, model_path: str):
        # Load model
        self.model = EmotionClassifier.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.to('mps')  # For Apple Silicon
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Model prediction function for SHAP."""
        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to('mps')

        # Get predictions
        with torch.no_grad():
            logits = self.model(encodings['input_ids'], encodings['attention_mask'])
            probs = torch.sigmoid(logits).cpu().numpy()
        
        return probs

    def explain_text(self, text: str) -> Dict[str, Any]:
        """Generate SHAP explanation for a single text."""
        # Create explainer
        explainer = shap.Explainer(
            self.predict_proba,
            self.tokenizer,
            output_names=self.emotion_labels
        )
        
        # Calculate SHAP values
        shap_values = explainer([text])
        
        # Get word attributions
        words = self.tokenizer.tokenize(text)
        attributions = {
            emotion: {
                word: value 
                for word, value in zip(words, values)
            }
            for emotion, values in zip(self.emotion_labels, shap_values[0].values)
        }
        
        return {
            'text': text,
            'attributions': attributions,
            'shap_values': shap_values
        }

    def explain_batch(self, texts: List[str], output_dir: str = 'explanations'):
        """Generate and save SHAP explanations for a batch of texts."""
        # Create explainer
        explainer = shap.Explainer(
            self.predict_proba,
            self.tokenizer,
            output_names=self.emotion_labels
        )
        
        # Calculate SHAP values
        shap_values = explainer(texts)
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save plots
        self._save_summary_plot(shap_values, texts, output_dir)
        self._save_emotion_plots(shap_values, texts, output_dir)
        
        return shap_values

    def _save_summary_plot(self, shap_values, texts: List[str], output_dir: str):
        """Save overall SHAP summary plot."""
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            texts,
            class_names=self.emotion_labels,
            show=False
        )
        plt.savefig(f'{output_dir}/shap_summary.png', bbox_inches='tight')
        plt.close()

    def _save_emotion_plots(self, shap_values, texts: List[str], output_dir: str):
        """Save individual emotion SHAP plots."""
        for i, emotion in enumerate(self.emotion_labels):
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values[:, :, i],
                texts,
                plot_type='bar',
                show=False
            )
            plt.title(f'SHAP Values for {emotion.capitalize()}')
            plt.savefig(f'{output_dir}/shap_{emotion}.png', bbox_inches='tight')
            plt.close()

def main():
    # Example usage
    explainer = EmotionExplainer('checkpoints/best_model.ckpt')
    
    # Load some test data
    test_data = pd.read_csv('data/semeval/test/eng.csv')
    sample_texts = test_data['text'].iloc[:100].tolist()
    
    # Generate explanations
    shap_values = explainer.explain_batch(sample_texts)
    
    # Example of single text explanation
    example_text = test_data['text'].iloc[0]
    explanation = explainer.explain_text(example_text)
    
    # Print example attribution values
    print("\nExample Text:", example_text)
    print("\nEmotion Attributions:")
    for emotion, word_scores in explanation['attributions'].items():
        print(f"\n{emotion.capitalize()}:")
        # Print top 5 words contributing to this emotion
        sorted_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for word, score in sorted_words:
            print(f"  {word}: {score:.3f}")

if __name__ == '__main__':
    main()
