import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    StochasticWeightAveraging
)
from pytorch_lightning.loggers import TensorBoardLogger
from data import EmotionDataModule
from model import ImprovedEmotionClassifier
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
import os
import torch

class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_f1s = []
        self.val_f1s = []
        self.learning_rates = []
        self.current_epoch = 0
        self.emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
        self.emotion_f1s = {emotion: [] for emotion in self.emotion_labels}
        self.steps = []
        self.current_step = 0

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_f1 = metrics.get('train_f1')
        val_f1 = metrics.get('val_f1')
        train_loss = metrics.get('train_loss_epoch')  # Changed from train_loss to train_loss_epoch
        val_loss = metrics.get('val_loss')
        
        # Store metrics only at epoch end
        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        if train_f1 is not None:
            self.train_f1s.append(train_f1.item())
        if val_loss is not None:
            self.val_losses.append(val_loss.item())
        if val_f1 is not None:
            self.val_f1s.append(val_f1.item())
        
        # Store current learning rate
        if pl_module.optimizers() is not None:
            self.learning_rates.append(
                pl_module.optimizers().param_groups[0]['lr']
            )
        
        print(f"\nEpoch {self.current_epoch} Summary:")
        if train_f1 is not None:
            print(f"Train F1: {train_f1.item():.4f}")
        if train_loss is not None:
            print(f"Train Loss: {train_loss.item():.4f}")
        if val_f1 is not None:
            print(f"Val F1: {val_f1.item():.4f}")
        if val_loss is not None:
            print(f"Val Loss: {val_loss.item():.4f}")
            
        # Print loss weights if available
        if hasattr(pl_module, 'loss_weights'):
            print("\nLoss weights:")
            for emotion, weight in zip(self.emotion_labels, pl_module.loss_weights):
                print(f"{emotion}: {weight.item():.4f}")
        
        print("-" * 50)
        self.current_epoch += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get('val_loss')
        val_f1 = metrics.get('val_f1')
        
        # Only store validation metrics at epoch end
        if self.current_step == len(self.val_losses):
            if val_loss is not None:
                self.val_losses.append(val_loss.item())
            if val_f1 is not None:
                self.val_f1s.append(val_f1.item())
            self.steps.append(self.current_step)
        self.current_step += 1

    def plot_metrics(self):
        os.makedirs('plots', exist_ok=True)
        
        # Use steps for x-axis instead of epochs
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        if self.train_losses:
            plt.plot(range(len(self.train_losses)), self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(range(len(self.val_losses)), self.val_losses, label='Val Loss')
        plt.title('Loss Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # F1 plot
        plt.subplot(2, 2, 2)
        if self.train_f1s:
            plt.plot(range(len(self.train_f1s)), self.train_f1s, label='Train F1')
        if self.val_f1s:
            plt.plot(range(len(self.val_f1s)), self.val_f1s, label='Val F1')
        plt.title('F1 Score Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot
        if self.learning_rates:
            plt.subplot(2, 2, 3)
            plt.plot(range(len(self.learning_rates)), self.learning_rates)
            plt.title('Learning Rate Evolution')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/training_metrics.png')
        plt.close()

def train():
    # Create directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Data module with memory-efficient settings
    data_module = EmotionDataModule(
        train_file='data/semeval/train/eng.csv',
        val_file='data/semeval/dev/eng.csv',
        test_file='data/semeval/test/eng.csv',
        batch_size=8,  # Small batch size
        max_length=128,
        num_workers=4
    )
    data_module.setup()
    
    # Print dataset statistics
    data_module.print_dataset_stats()
    
    # Initialize model
    model = ImprovedEmotionClassifier(
        learning_rate=1e-5,
        warmup_steps=100,
        class_weights=data_module.class_weights
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_f1',
            dirpath='checkpoints',
            filename='emotion-{epoch:02d}-{val_f1:.2f}',
            save_top_k=3,
            mode='max',
            save_weights_only=True
        ),
        EarlyStopping(
            monitor='val_f1',
            patience=10,           # Increased from 5 to 10
            mode='max',
            min_delta=0.001,      # Keep small improvement threshold
            check_finite=True,    # Ensure we're not getting NaN values
            check_on_train_epoch_end=False,  # Only check on validation
            stopping_threshold=0.95,  # Stop if we reach 95% F1
            divergence_threshold=0.0  # Don't stop for divergence
        ),
        LearningRateMonitor(logging_interval='step'),
        StochasticWeightAveraging(
            swa_epoch_start=0.8,
            swa_lrs=1e-5/2
        ),
        MetricsCallback()
    ]
    
    # Logger
    logger = TensorBoardLogger('logs', name='emotion_detection')
    
    # Trainer with robust settings
    trainer = pl.Trainer(
        accelerator='mps',
        devices=1,
        max_epochs=30,  # Increased from 20
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,  # Increased from 0.5
        accumulate_grad_batches=8,  # Increased from 4
        val_check_interval=0.5,  # Reduced from 0.25 for stability
        precision=32,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=2,
        log_every_n_steps=10
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Plot final metrics
    metrics_callback = next(cb for cb in callbacks if isinstance(cb, MetricsCallback))
    metrics_callback.plot_metrics()
    
    # Test
    trainer.test(model, data_module)
    
    return model, trainer

if __name__ == '__main__':
    train()