import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import RobertaModel, RobertaConfig
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall
)
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

class EmotionClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = 'roberta-base',
        num_labels: int = 5,
        learning_rate: float = 2e-5,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load RoBERTa configuration and model
        self.config = RobertaConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        
        # Add special tokens for emotions
        self.special_token_ids = {
            'anger': '[ANGER]',
            'fear': '[FEAR]',
            'joy': '[JOY]',
            'sadness': '[SADNESS]',
            'surprise': '[SURPRISE]'
        }
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, num_labels)
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.train_metrics = MetricCollection({
            'accuracy': MultilabelAccuracy(num_labels=num_labels),
            'f1': MultilabelF1Score(num_labels=num_labels),
            'precision': MultilabelPrecision(num_labels=num_labels),
            'recall': MultilabelRecall(num_labels=num_labels)
        })
        
        self.val_metrics = self.train_metrics.clone(prefix='val_')
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Update metrics
        preds = torch.sigmoid(logits)
        self.train_metrics.update(preds, labels)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log_dict(self.train_metrics, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Update metrics
        preds = torch.sigmoid(logits)
        self.val_metrics.update(preds, labels)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.val_metrics, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Create optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Create scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
    
    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        logits = self(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).float()
        
        return {
            'probabilities': probs,
            'predictions': predictions
        }
    
    def on_train_epoch_end(self):
        # Compute and log epoch-level metrics
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.train_metrics.reset()
        
    def on_validation_epoch_end(self):
        # Compute and log epoch-level metrics
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()
        
    def save_predictions(self, predictions, output_file):
        """Save predictions to file"""
        np.save(output_file, predictions)
        
    def visualize_predictions(self, text, probs, predictions):
        """Visualize model predictions for a single text"""
        emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
        for emotion, prob, pred in zip(emotion_labels, probs, predictions):
            print(f"{emotion}: {prob:.3f} ({'✓' if pred else '✗'})")