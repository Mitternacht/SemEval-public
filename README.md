# SemEval 2025 Task 11-A: Multi-Label Emotion Detection

## Project Overview
This project implements a multi-label emotion detection system for SemEval 2025 Task 11-A. The system predicts five emotions (anger, fear, joy, sadness, surprise) from text input using RoBERTa with enhanced attention mechanisms.

### Task Description
Given a text snippet, predict the perceived emotion(s) of the speaker. This is a multi-label classification task where:
- Multiple emotions can be present simultaneously
- Target emotions: anger, fear, joy, sadness, surprise
- Evaluation metric: F1-score (target >90%)

## Implementation Details

### Project Structure
```
src/
├── data.py           # Data loading and preprocessing
├── download_data.py  # Dataset download and preparation
├── evaluate.py       # Model evaluation and metrics
├── explain.py        # SHAP-based model explanations
├── model.py          # Enhanced RoBERTa model implementation
└── train.py         # Training configuration and execution
```

### Data Sources
- SemEval dataset (train-eng.csv, dev-eng.csv)
- GoEmotions dataset (HuggingFace) - https://huggingface.co/datasets/google-research-datasets/go_emotions
- DAIR-AI Emotion dataset (HuggingFace) - https://huggingface.co/datasets/dair-ai/emotion

### Model Architecture

#### Base Model
- RoBERTa-base
- 12 attention layers
- 768 hidden dimension
- 125M parameters

#### Enhanced Components
1. Emotion-specific Processing:
   - Individual attention for each emotion
   - Emotion-specific tokens
   - Dedicated feature extractors

2. Training Optimizations:
   - Mixed precision training
   - Gradient checkpointing
   - Gradient accumulation
   - Layer freezing

### Data Processing Features
- Multi-source dataset combination
- Automatic data balancing
- Text augmentation (word swap, deletion)
- Emoji removal
- Label standardization
- Caching system for tokenized data

### Training Configuration
- Optimizer: AdamW
- Learning rate: 2e-5
- Batch size: 32 (configurable)
- Early stopping with F1 monitoring
- TensorBoard logging
- Mixed precision training support
- Gradient clipping

## Usage

### Data Preparation
```bash
python src/download_data.py
```

### Training
```bash
python src/train.py \
    --data_dir data \
    --batch_size 32 \
    --max_epochs 10 \
    --learning_rate 2e-5 \
    --use_fp16
```

### Evaluation
```bash
python src/evaluate.py \
    --checkpoint_path checkpoints/best_model.ckpt \
    --plot_confusion \
    --output_predictions
```

### Model Explanations
```bash
python src/explain.py \
    --model_path checkpoints/best_model.ckpt \
    --text "your text here" \
    --output_dir results/shap
```

## Model Explainability
The project includes comprehensive model explanation capabilities:
- SHAP analysis for predictions
- Attention pattern visualization
- Token-level contribution analysis
- Per-emotion explanation plots

## Hardware Configuration
- CUDA-capable GPU
- 64GB RAM recommended
- CUDA toolkit 11.8+
- cuDNN 8.7+

## Software Dependencies
- PyTorch 2.5.1
- PyTorch Lightning 2.5.0
- Transformers 4.48.1
- numpy 1.26.4
- pandas 2.2.3
- scikit-learn 1.5.2
- SHAP 0.46
- emoji 2.2.0
- datasets 2.12.0

## Performance Monitoring
- Overall F1 score tracking
- Per-emotion metrics
- Loss monitoring
- Learning rate scheduling
- Validation metrics
- Confusion matrix visualization

## Output Files
- Model checkpoints
- TensorBoard logs
- Prediction CSVs
- SHAP visualizations
- Confusion matrices
- Token importance plots

## Future Improvements
- Ensemble methods investigation
- Advanced data augmentation
- Cross-lingual support
- Performance optimization
- Additional explainability methods

## License
This project is part of the SemEval 2025 competition. Please refer to the competition guidelines for usage rights and restrictions.