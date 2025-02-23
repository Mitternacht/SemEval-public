# SemEval 2025 Task 11-A: Multi-Label Emotion Detection

## Project Overview
This project implements a multi-label emotion detection system for SemEval 2025 Task 11-A. The system predicts five emotions (anger, fear, joy, sadness, surprise) from text input using RoBERTa.

### Task Description
Given a text snippet, predict the perceived emotion(s) of the speaker. This is a multi-label classification task where:
- Multiple emotions can be present simultaneously
- Target emotions: anger, fear, joy, sadness, surprise
- Evaluation metric: F1-score (target >90%)

## Implementation Details

### Project Structure
```
src/
├── data.py           # Dataset and DataLoader implementations
├── download_data.py  # Multi-source dataset processing and combination
├── evaluate.py       # Model evaluation with detailed metrics
├── explain.py       # SHAP and attention-based explanations
├── human_eval.py    # Human evaluation comparison tools
├── model.py         # RoBERTa-based classifier implementation
└── train.py         # Training configuration and execution
```

### Data Sources
- SemEval dataset (train-eng.csv, dev-eng.csv, test-eng.csv)
- GoEmotions dataset (HuggingFace)
- DAIR-AI Emotion dataset (HuggingFace)

### Model Architecture
- Base Model: RoBERTa-base
- Classification Head:
  - Dropout (0.1)
  - Linear layer (768 -> 5)
- Special emotion tokens added to vocabulary
- Binary Cross-Entropy loss with logits

### Data Processing Features
- Multi-dataset merging and cleaning
- Automated data balancing
- Text cleaning (emoji removal, whitespace normalization)
- Efficient caching system for tokenized data
- Balanced sampling with sample weights
- Maximum sequence length: 128 tokens

### Training Configuration
- Optimizer: AdamW
- Learning rate: 2e-5
- Weight decay: 0.01
- Warmup steps: 1000
- Batch size: 32
- Maximum epochs: 10
- Early stopping (patience=3, min_delta=0.001)
- Gradient clipping: 1.0
- Mixed precision training (optional)

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
    --learning_rate 2e-5 \
    --max_epochs 10 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
```

### Evaluation
```bash
python src/evaluate.py \
    --data_path data/semeval/test/eng.csv \
    --checkpoint_path checkpoints/model.ckpt \
    --batch_size 32 \
    --plot_confusion \
    --output_predictions
```

### Model Explanations
```bash
python src/explain.py \
    --model_path checkpoints/model.ckpt \
    --text "your text here" \
    --output_dir results/shap
```

### Human Evaluation
```bash
python src/human_eval.py \
    --test_data data/semeval/test/eng.csv \
    --model_checkpoint checkpoints/model.ckpt \
    --collect_annotations \
    --annotator_id A1
```

## Metrics & Monitoring
- Per-emotion metrics:
  - F1 Score
  - Precision
  - Recall
  - Accuracy
- Training monitoring:
  - TensorBoard logging
  - Loss tracking
  - Learning rate scheduling
  - Validation metrics

## Output Files
- Model checkpoints (top 3 + latest)
- TensorBoard logs
- Evaluation results:
  - Confusion matrices
  - Detailed predictions CSV
  - Classification reports
- SHAP visualizations:
  - Token importance plots
  - Per-emotion explanations
- Human evaluation comparisons

## Hardware Configuration
- NVIDIA GPU with CUDA support
- 64GB RAM recommended
- CUDA toolkit 11.8
- cuDNN 8.7

## Software Dependencies
Key dependencies (see description.md for full list):
- PyTorch 2.5.1
- PyTorch Lightning 2.5.0
- Transformers 4.48.1
- SHAP 0.46
- numpy 1.26.4
- pandas 2.2.3
- scikit-learn 1.5.2
- datasets 2.12.0

## License
This project is part of the SemEval 2025 competition. Please refer to the competition guidelines for usage rights and restrictions.
