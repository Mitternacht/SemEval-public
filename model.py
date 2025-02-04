import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import RobertaModel
from torchmetrics.classification import MultilabelF1Score
import torch.nn.functional as F

class EmotionClassifier(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-5,
        warmup_steps: int = 100,
        class_weights: torch.Tensor = None,
        hidden_size: int = 1024,      
        dropout_rate: float = 0.2,
        num_attention_heads: int = 16
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # TUNE:
        # - 'roberta-base'
        # - 'microsoft/deberta-v3-large'
        self.roberta = RobertaModel.from_pretrained('roberta-large', add_pooling_layer=False)
        
        # Emotion-specific attention
        self.emotion_attention = nn.MultiheadAttention(
            hidden_size, 
            num_attention_heads,
            dropout=dropout_rate
        )
        
        # TUNE: Try different classifier architectures
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(hidden_size//2, 5)
        )
        
        # Metrics
        self.train_f1 = MultilabelF1Score(
            num_labels=5, 
            threshold=0.5,
            validate_args=True,
            average='macro'
        )
        self.val_f1 = MultilabelF1Score(
            num_labels=5, 
            threshold=0.5,
            validate_args=True,
            average='macro'
        )
        self.test_f1 = MultilabelF1Score(
            num_labels=5, 
            threshold=0.5,
            validate_args=True,
            average='macro'
        )
        
        self.class_weights = class_weights

    def forward(self, input_ids, attention_mask):
        # RoBERTa encoding
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Emotion-specific attention
        attended_output, _ = self.emotion_attention(
            sequence_output.permute(1, 0, 2),
            sequence_output.permute(1, 0, 2),
            sequence_output.permute(1, 0, 2),
            key_padding_mask=~attention_mask.bool()
        )
        attended_output = attended_output.permute(1, 0, 2)
        
        # Pool the attended outputs
        pooled_output = torch.mean(attended_output, dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].float()
        
        logits = self(input_ids, attention_mask)
        
        # Add epsilon to prevent log(0)
        eps = 1e-7
        preds = torch.sigmoid(logits.float())
        preds = torch.clamp(preds, eps, 1 - eps)
        
        loss = F.binary_cross_entropy_with_logits(
            logits.float(), 
            labels,
            pos_weight=self.class_weights.to(self.device) if self.class_weights is not None else None,
            reduction='mean'
        )
        
        # Update F1 metric
        self.train_f1.update(preds, labels)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def on_train_epoch_end(self):
        # Compute F1 at epoch end
        train_f1 = self.train_f1.compute()
        self.log('train_f1', train_f1, prog_bar=True)
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].float()
        
        logits = self(input_ids, attention_mask)
        
        eps = 1e-7
        preds = torch.sigmoid(logits.float())
        preds = torch.clamp(preds, eps, 1 - eps)
        
        loss = F.binary_cross_entropy_with_logits(
            logits.float(), 
            labels,
            reduction='mean'
        )
        
        # Update F1 metric
        self.val_f1.update(preds, labels)
        
        # Log loss
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        return loss

    def on_validation_epoch_end(self):
        # Compute F1 at epoch end
        val_f1 = self.val_f1.compute()
        self.log('val_f1', val_f1, prog_bar=True, sync_dist=True)
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        preds = torch.sigmoid(logits)
        f1 = self.test_f1(preds, labels)
        
        self.log('test_f1', f1)
        
        return f1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.1
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
