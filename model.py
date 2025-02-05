import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import RobertaModel
from torchmetrics.classification import MultilabelF1Score
import torch.nn.functional as F
import numpy as np

class ImprovedEmotionClassifier(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-5,
        warmup_steps: int = 100,
        class_weights: torch.Tensor = None,
        hidden_size: int = 1024,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # RoBERTa with gradient checkpointing
        self.roberta = RobertaModel.from_pretrained(
            'roberta-large',
            add_pooling_layer=False,
            use_cache=False
        )
        self.roberta.gradient_checkpointing_enable()
        
        # Freeze embeddings and first 16 layers
        self.freeze_layers(16)
        
        # Weighted layer pooling
        self.layer_weights = nn.Parameter(torch.ones(24) / 24)
        
        # Emotion-specific attention per class
        self.emotion_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
                nn.Softmax(dim=1)
            ) for _ in range(5)
        ])
        
        # Emotion-specific classifiers
        self.emotion_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.Dropout(dropout_rate),
                nn.GELU(),
                nn.Linear(hidden_size // 2, 1)
            ) for _ in range(5)
        ])
        
        # Loss weights for dynamic balancing
        self.register_buffer('loss_weights', torch.ones(5))
        self.loss_history = {i: [] for i in range(5)}
        
        # Metrics
        self.train_f1 = MultilabelF1Score(num_labels=5)
        self.val_f1 = MultilabelF1Score(num_labels=5)
        self.test_f1 = MultilabelF1Score(num_labels=5)
        self.class_weights = class_weights

    def freeze_layers(self, num_layers):
        # Freeze embeddings
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
            
        # Freeze first num_layers
        for layer in self.roberta.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Ensure correct types
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        
        # Get all hidden states
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Weighted sum of last 8 layers
        last_hidden_states = torch.stack(outputs.hidden_states[-8:], dim=-1)
        layer_weights = F.softmax(self.layer_weights[-8:], dim=0)
        weighted_hs = (last_hidden_states * layer_weights).sum(dim=-1)
        
        # Apply emotion-specific attention and classification
        logits = []
        for emotion_attn, emotion_clf in zip(self.emotion_attentions, self.emotion_classifiers):
            # Compute attention weights
            attn_weights = emotion_attn(weighted_hs)
            
            # Apply attention
            emotion_context = torch.bmm(
                attn_weights.transpose(1, 2),
                weighted_hs
            ).squeeze(1)
            
            # Classify
            emotion_logit = emotion_clf(emotion_context)
            logits.append(emotion_logit)
        
        return torch.cat(logits, dim=1)

    def _compute_loss(self, logits, labels):
        # Label smoothing
        smoothing = 0.1
        labels = labels * (1 - smoothing) + 0.5 * smoothing
        
        # Compute per-emotion losses
        losses = []
        for i in range(5):
            emotion_loss = F.binary_cross_entropy_with_logits(
                logits[:, i],
                labels[:, i],
                pos_weight=self.class_weights[i].to(self.device) if self.class_weights is not None else None
            )
            losses.append(emotion_loss)
            
            # Update loss history
            self.loss_history[i].append(emotion_loss.item())
            if len(self.loss_history[i]) > 100:  # Keep last 100 steps
                self.loss_history[i].pop(0)
        
        # Update loss weights using moving averages
        with torch.no_grad():
            avg_losses = torch.tensor([sum(hist) / len(hist) for hist in self.loss_history.values()])
            max_loss = avg_losses.max()
            self.loss_weights = max_loss / avg_losses
        
        # Compute weighted average loss
        total_loss = sum(loss * weight for loss, weight in zip(losses, self.loss_weights)) / len(losses)
        
        return total_loss

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].long()
        attention_mask = batch['attention_mask'].long()
        labels = batch['labels'].float()
        
        # Store original labels for metrics
        original_labels = labels.clone()
        
        # Mixup augmentation
        if torch.rand(1) < 0.5:  # 50% chance of mixup
            lambda_mix = torch.distributions.beta.Beta(0.2, 0.2).sample()
            index = torch.randperm(input_ids.size(0))
            
            mixed_input_ids = input_ids[index]
            mixed_attention_mask = attention_mask[index]
            mixed_labels = labels[index]
            
            # Mix the inputs and labels
            input_ids = input_ids  # Keep original input_ids (discrete tokens)
            attention_mask = attention_mask | mixed_attention_mask  # Union of attention masks
            labels = lambda_mix * labels + (1 - lambda_mix) * mixed_labels  # Mixed labels for loss
        
        logits = self(input_ids, attention_mask)
        loss = self._compute_loss(logits, labels)
        
        # Log metrics using original binary labels
        preds = torch.sigmoid(logits)
        self.train_f1.update(preds, original_labels)  # Use original binary labels for F1
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].float()
        
        logits = self(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        
        # Update F1 metric
        preds = torch.sigmoid(logits)
        self.val_f1.update(preds, labels)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        return loss

    def on_train_epoch_end(self):
        train_f1 = self.train_f1.compute()
        self.log('train_f1', train_f1, prog_bar=True)
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        val_f1 = self.val_f1.compute()
        self.log('val_f1', val_f1, prog_bar=True, sync_dist=True)
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        """Test step for model evaluation."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].float()
        
        logits = self(input_ids, attention_mask)
        preds = torch.sigmoid(logits)
        
        # Only compute metrics if we have valid labels (non-zero tensors)
        if labels.sum() > 0:
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
            self.test_f1.update(preds, labels)
            self.log('test_loss', loss, prog_bar=True, sync_dist=True)
            return loss
        
        # For prediction-only mode (test set without labels)
        return {
            'predictions': preds,
            'text_ids': batch['input_ids']  # Can be used to map back to original texts
        }

    def on_test_epoch_end(self):
        """Called at the end of test to compute final metrics."""
        test_f1 = self.test_f1.compute()
        self.log('test_f1', test_f1, prog_bar=True, sync_dist=True)
        self.test_f1.reset()
        
        print(f"\nTest Results:")
        print(f"Test F1: {test_f1:.4f}")

    def configure_optimizers(self):
        # Differential learning rates
        no_decay = ['bias', 'LayerNorm.weight']
        
        # Parameters with different learning rates and weight decay
        optimizer_grouped_parameters = []
        
        # Lower learning rate for frozen layers (if you want to fine-tune them later)
        for layer in self.roberta.encoder.layer[:16]:
            optimizer_grouped_parameters.append({
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
                "lr": self.hparams.learning_rate * 0.1
            })
            optimizer_grouped_parameters.append({
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.hparams.learning_rate * 0.1
            })
        
        # Higher learning rate for top layers
        for layer in self.roberta.encoder.layer[16:]:
            optimizer_grouped_parameters.append({
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
                "lr": self.hparams.learning_rate
            })
            optimizer_grouped_parameters.append({
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.hparams.learning_rate
            })
        
        # Highest learning rate for emotion-specific layers
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in self.named_parameters() 
                          if "emotion" in n and not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
                "lr": self.hparams.learning_rate * 1.5
            },
            {
                "params": [p for n, p in self.named_parameters() 
                          if "emotion" in n and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.hparams.learning_rate * 1.5
            }
        ])
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer_grouped_parameters],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos',
            cycle_momentum=False
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }