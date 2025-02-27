import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from transformers import RobertaTokenizer
from model import EmotionClassifier
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import torch.nn as nn
import pandas as pd

class EmotionExplainer:
    def __init__(self, model_path: str):
        """Initialize explainer with trained model."""
        # Load model and tokenizer
        self.model = EmotionClassifier.load_from_checkpoint(model_path)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Initialize RoBERTa tokenizer with emotion tokens
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
        special_tokens = [f'[{e.upper()}]' for e in self.emotion_labels]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    def explain_text(self, text: str, output_path: str = None) -> Dict:
        """Generate explanation for a given text."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Move inputs to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions and attention weights
        with torch.no_grad():
            outputs = self.model.roberta(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_attentions=True
            )
            logits = self.model.classifier(outputs.pooler_output)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
        
        # Get attention from last layer
        attention = outputs.attentions[-1]
        
        # Process attention weights
        token_importances = self._process_attention(
            attention,
            inputs['input_ids'][0],
            inputs['attention_mask'][0]
        )
        
        # Move predictions back to CPU for numpy conversion
        probs = probs.cpu()
        preds = preds.cpu()
        
        # Visualize if requested
        if output_path:
            self._visualize_explanation(
                text,
                token_importances,
                probs.squeeze().numpy(),
                preds.squeeze().numpy(),
                output_path
            )
        
        return {
            'text': text,
            'probabilities': probs.squeeze().numpy(),
            'predictions': preds.squeeze().numpy(),
            'token_importances': token_importances
        }
    
    def _process_attention(
        self,
        attention: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[Tuple[str, float]]:
        """Process attention weights to get token importances."""
        # Average attention across heads and layers
        # Shape: [batch_size, num_heads, seq_len, seq_len] -> [seq_len]
        mean_attention = attention.mean(dim=(0, 1))  # Average across batch and heads
        mean_attention = mean_attention.mean(dim=0)   # Average across sequence dimension
        
        # Get token importances
        token_importances = []
        input_ids = input_ids.cpu().numpy()  # Convert to numpy array
        attention_mask = attention_mask.cpu().numpy()
        
        # Process only the valid tokens (where attention_mask is 1)
        valid_length = attention_mask.sum()
        
        for i in range(valid_length):
            # Get token and its importance score
            token = self.tokenizer.convert_ids_to_tokens(int(input_ids[i]))  # Convert to int
            # Clean the Ġ character from token
            token = token.replace('Ġ', '')
            score = mean_attention[i].item()  # Now this should be a scalar
            
            # Skip special tokens
            if token not in ['<s>', '</s>', '<pad>']:
                token_importances.append((token, score))
        
        return token_importances
    
    def _visualize_explanation(
        self,
        text: str,
        token_importances: List[Tuple[str, float]],
        probabilities: np.ndarray,
        predictions: np.ndarray,
        output_path: str
    ):
        """Create visualization of the explanation."""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
        
        # Plot token importances
        tokens, scores = zip(*token_importances)
        data = pd.DataFrame({
            'Token': range(len(tokens)),
            'Score': scores
        })
        sns.barplot(
            data=data,
            x='Token',
            y='Score',
            ax=ax1,
            legend=False
        )
        ax1.set_xticks(range(len(tokens)))
        ax1.set_xticklabels(tokens, rotation=45, ha='right')
        ax1.set_title('Token Importance Scores')
        
        # Plot emotion probabilities
        data = pd.DataFrame({
            'Emotion': self.emotion_labels,
            'Probability': probabilities,
            'Status': ['Predicted' if p else 'Not Predicted' for p in predictions]
        })
        sns.barplot(
            data=data,
            x='Emotion',
            y='Probability',
            hue='Status',
            palette={'Predicted': 'green', 'Not Predicted': 'red'},
            ax=ax2,
            legend=False
        )
        ax2.set_title('Emotion Probabilities')
        ax2.set_ylim(0, 1)
        
        # Add text predictions
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            color = 'green' if pred else 'red'
            ax2.text(i, prob, f'{prob:.2f}', ha='center', va='bottom', color=color)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def explain_with_shap(self, text: str, output_dir: str = 'results/shap'):
        """Generate SHAP explanations for the input text."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get tokens for visualization and clean them
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        tokens = [t.replace('Ġ', '') for t in tokens]
        
        # Create prediction function for SHAP that returns values for all emotions
        def f(x):
            if isinstance(x, str):
                x = [x]
            x = np.array(x)
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.long, device=self.device)
                attention_mask = torch.ones_like(x_tensor, dtype=torch.long)
                output = self.model(x_tensor, attention_mask)
                # Return individual probabilities for each emotion
                probs = torch.sigmoid(output).cpu().numpy()
                return probs if len(probs.shape) == 2 else probs.reshape(1, -1)
        
        # Create more diverse background data
        n_background = 10
        background_data = []
        for _ in range(n_background):
            random_ids = torch.randint(0, self.tokenizer.vocab_size, (1, inputs['input_ids'].shape[1]))
            background_data.append(random_ids[0].numpy())
        background_data = np.array(background_data)
        
        # Initialize SHAP explainer with more samples
        explainer = shap.KernelExplainer(
            f,
            background_data,
            link="identity"
        )
        
        # Generate SHAP values with more samples
        input_data = inputs['input_ids'][0].cpu().numpy()
        try:
            # Try to get SHAP values for all outputs at once
            shap_values = explainer.shap_values(
                input_data,
                nsamples=500,
                l1_reg="num_features(10)"
            )
            
            # Print shape information for debugging
            print(f"SHAP values type: {type(shap_values)}")
            if isinstance(shap_values, list):
                print(f"Number of SHAP value arrays: {len(shap_values)}")
                print(f"Shape of first SHAP array: {shap_values[0].shape}")
            else:
                print(f"Shape of SHAP values: {shap_values.shape}")
            
        except Exception as e:
            print(f"Error generating SHAP values: {e}")
            # Fallback: generate SHAP values for each emotion separately
            shap_values = []
            for i in range(len(self.emotion_labels)):
                emotion_explainer = shap.KernelExplainer(
                    lambda x: f(x)[:, i],
                    background_data
                )
                emotion_shap = emotion_explainer.shap_values(input_data, nsamples=100)
                shap_values.append(emotion_shap)
        
        # Ensure we have the right format
        if not isinstance(shap_values, list):
            shap_values = [shap_values[:, i] for i in range(shap_values.shape[1])]
        
        # Plot explanations for each emotion
        plt.figure(figsize=(20, 15))
        
        for idx, emotion in enumerate(self.emotion_labels):
            plt.subplot(3, 2, idx + 1)
            
            # Get values for this emotion and ensure it's 1D and the right length
            values = np.array(shap_values[idx])
            if len(values.shape) > 1:
                values = values.mean(axis=0)  # Take mean if we have multiple samples
            
            # Ensure values matches the number of tokens
            if len(values) > len(tokens):
                values = values[:len(tokens)]
            elif len(values) < len(tokens):
                values = np.pad(values, (0, len(tokens) - len(values)))
                
            x_positions = np.arange(len(tokens))
            
            # Create bar plot
            bars = plt.bar(x_positions, values)
            
            # Color bars based on value
            for bar, value in zip(bars, values):
                if value > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            # Set x-ticks and labels
            plt.xticks(x_positions, tokens, rotation=45, ha='right')
            plt.title(f'SHAP Values for {emotion}')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on top of bars
            for i, v in enumerate(values):
                plt.text(i, v, f'{v:.2f}', 
                        ha='center', 
                        va='bottom' if v >= 0 else 'top',
                        fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_explanation.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Get predictions with no gradient computation
        with torch.no_grad():
            logits = self.model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs > 0.5).astype(np.float32)
        
        return {
            'shap_values': shap_values,
            'probabilities': probs,
            'predictions': preds,
            'tokens': tokens
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--text', type=str, required=True, help='Text to explain')
    parser.add_argument('--output_dir', type=str, default='results/shap')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize explainer
    explainer = EmotionExplainer(args.model_path)
    
    # Generate and save attention-based explanation
    attention_path = Path(args.output_dir) / 'attention_explanation.png'
    explanation = explainer.explain_text(args.text, str(attention_path))
    
    # Generate SHAP explanations
    shap_explanation = explainer.explain_with_shap(args.text, args.output_dir)
    
    # Print predictions
    print("\nPredictions:")
    for emotion, (prob, pred) in zip(explainer.emotion_labels, 
                                    zip(explanation['probabilities'], 
                                        explanation['predictions'])):
        status = "✓" if pred else "✗"
        print(f"{emotion}: {prob:.3f} {status}")
    
    print(f"\nExplanation visualizations saved to:")
    print(f"- Attention: {attention_path}")
    print(f"- SHAP: {args.output_dir}/shap_explanation.png")

if __name__ == "__main__":
    main()