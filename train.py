import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from data import EmotionDataModule
from model import EmotionClassifier
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
import os

class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_f1s = []
        self.val_f1s = []
        self.learning_rates = []
        self.current_epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_f1 = metrics.get('train_f1')
        val_f1 = metrics.get('val_f1')
        train_loss = metrics.get('train_loss')
        val_loss = metrics.get('val_loss')
        
        # Store metrics (handle None values)
        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        if train_f1 is not None:
            self.train_f1s.append(train_f1.item())
        
        self.learning_rates.append(
            pl_module.optimizers().param_groups[0]['lr']
        )
        
        # Print epoch summary with safe formatting
        print(f"\nEpoch {self.current_epoch} Summary:")
        if train_f1 is not None:
            print(f"Train F1: {train_f1.item():.4f}")
        if train_loss is not None:
            print(f"Train Loss: {train_loss.item():.4f}")
        if val_f1 is not None:
            print(f"Val F1: {val_f1.item():.4f}")
        if val_loss is not None:
            print(f"Val Loss: {val_loss.item():.4f}")
        print("-" * 50)
        
        self.current_epoch += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get('val_loss')
        val_f1 = metrics.get('val_f1')
        
        # Store metrics (handle None values)
        if val_loss is not None:
            self.val_losses.append(val_loss.item())
        if val_f1 is not None:
            self.val_f1s.append(val_f1.item())

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)
        
        # Plot losses
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Val Loss')
        plt.title('Loss Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot F1 scores
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_f1s, label='Train F1')
        plt.plot(epochs, self.val_f1s, label='Val F1')
        plt.title('F1 Score Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        # Plot learning rate
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.learning_rates)
        plt.title('Learning Rate Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.savefig('plots/training_metrics.png')
        plt.close()

def train():
    # Create directories for outputs
    os.makedirs('plots', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Clear existing checkpoints (optional)
    # import shutil
    # shutil.rmtree(checkpoint_dir)
    # os.makedirs(checkpoint_dir)

    data_module = EmotionDataModule(
        train_file='data/semeval/train/eng.csv',
        val_file='data/semeval/dev/eng.csv',
        test_file='data/semeval/test/eng.csv',
        batch_size=16,
        max_length=128
    )
    data_module.setup()
    
    # Print dataset statistics
    data_module.print_dataset_stats()

    # Model
    model = EmotionClassifier(
        learning_rate=1e-5,
        warmup_steps=100,
        class_weights=data_module.class_weights
    )

    # Callbacks
    metrics_callback = MetricsCallback()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        dirpath='checkpoints',
        filename='emotion-{epoch:02d}-{val_f1:.2f}',
        save_top_k=3,
        mode='max'
    )
    early_stopping = EarlyStopping(
        monitor='val_f1',
        patience=3,
        mode='max'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Logger
    logger = TensorBoardLogger('logs', name='emotion_detection')

    # TUNE: Training parameters
    trainer = pl.Trainer(
        accelerator='mps',
        devices=1,
        max_epochs=10,
        callbacks=[
            checkpoint_callback, 
            early_stopping, 
            lr_monitor,
            metrics_callback
        ],
        logger=logger,
        gradient_clip_val=1.0,
        precision=32,  # Change to full precision for stability
        accumulate_grad_batches=2,
        #val_check_interval=0.5
    )

    # Train
    trainer.fit(model, data_module)
    
    # Plot training metrics
    metrics_callback.plot_metrics()

    # Test
    trainer.test(model, data_module)

if __name__ == '__main__':
    train()
