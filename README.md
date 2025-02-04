# SemEval 2025 Task 11-A: Multi-Label Emotion Detection

## Project Overview
This project implements a multi-label emotion detection system for SemEval 2025 Task 11-A. The system predicts five emotions (anger, fear, joy, sadness, surprise) from text input using an enhanced RoBERTa-large model with emotion-specific attention mechanisms.

## Task Description
Given a text snippet, the system predicts which emotions are present. This is a multi-label classification task where:
- Multiple emotions can be present in a single text
- Emotions often co-occur in specific patterns
- Label distribution shows significant class imbalance

### Dataset Characteristics
- Training samples: 2,768
- Label distribution:
  - Single label: 41.2%
  - Two labels: 37.2%
  - Three labels: 10.8%
  - Four labels: 2.1%
  - Five labels: 0.1%
- Emotion distribution:
  - Anger: 12.0%
  - Fear: 58.2%
  - Joy: 24.3%
  - Sadness: 31.7%
  - Surprise: 30.3%
- Text characteristics:
  - Mean length: 20.5 tokens
  - Max length: 109 tokens

## Model Architecture

### Base Model
- RoBERTa-large
- 24 attention layers
- 1024 hidden dimension
- 355M parameters

### Enhanced Components
1. Emotion-specific Processing:
   - Individual attention heads for each emotion
   - Dedicated feature extractors per emotion
   - Emotion-specific classifiers

2. Co-occurrence Modeling:
   - Explicit co-occurrence layer
   - Combined feature processing
   - Learnable emotion interactions

3. Classification Head:
   ```python
   - Emotion-specific features (512 dim)
   - Co-occurrence layer (512 * 5 -> 512)
   - Individual emotion classifiers
   ```

## Implementation Details

### Files Structure
- `data.py`: Data loading and preprocessing
- `model.py`: Enhanced RoBERTa model implementation
- `train.py`: Training configuration and execution
- `evaluate.py`: Model evaluation and metrics
- `explain.py`: SHAP-based model explanations

### Data Processing
- RoBERTa tokenizer configuration
- Maximum sequence length: 128
- Class weight calculation for imbalance
- Multi-label encoding

### Training Configuration
- Optimizer: AdamW
- Learning rate: 1e-5 (base), 5e-5 (custom layers)
- Batch size: 16
- Gradient accumulation steps: 2
- Mixed precision training
- Early stopping on validation F1

### Loss Function
- Binary cross-entropy for each emotion
- Class weights for imbalance handling
- Individual loss terms per emotion

## Results Tracking
- Overall F1 score
- Per-emotion F1 scores
- Individual emotion losses
- Learning rate monitoring
- Validation metrics

## Model Explainability
- SHAP analysis for predictions
- Attention pattern visualization
- Per-emotion contribution analysis

## Hardware Requirements
- Apple M1 Max with 32GB RAM
- MPS acceleration support

## Software Dependencies
- PyTorch 2.5.1
- PyTorch Lightning 2.5.0
- Transformers 4.48.1
- numpy 1.26.4
- pandas 2.2.3
- scikit-learn 1.5.2
- SHAP 0.46

## Usage

### Training
```bash
python train.py
```

### Evaluation
```bash
python evaluate.py --model_path checkpoints/best_model.ckpt --test_file data/test.csv
```

### Generate Explanations
```bash
python explain.py --model_path checkpoints/best_model.ckpt --text "your text here"
```

## Model Deployment

### Loading the Model
```python
from model import EmotionClassifier

model = EmotionClassifier.load_from_checkpoint('checkpoints/best_model.ckpt')
model.eval()
```

### Making Predictions
```python
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
outputs = model(inputs['input_ids'], inputs['attention_mask'])
predictions = torch.sigmoid(outputs) > 0.5
```

## Performance Notes
- Current implementation focuses on emotion co-occurrence patterns
- Enhanced attention mechanisms for emotion-specific features
- Careful handling of class imbalance
- Model size vs performance tradeoffs considered
- F1 around 75


## Future Improvements
- Hyperparameter optimization
- Ensemble methods investigation
- More sophisticated co-occurrence modeling
- Advanced regularization techniques

