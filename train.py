import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import logging
from data import EmotionDataModule
from model import EmotionClassifier
import torch
import argparse

# Add Tensor Core optimization
torch.set_float32_matmul_precision('high')

def train(args):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize TensorBoard logger instead of wandb
    logger = TensorBoardLogger(
        save_dir='logs',
        name=args.run_name,
        default_hp_metric=False
    )
    
    # Initialize data module
    data_module = EmotionDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = EmotionClassifier(
        model_name=args.model_name,
        num_labels=5,  # Our 5 emotions
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay
    )
    
    # Add F1 specific callback
    class F1LoggingCallback(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            
            # Fix the metric name to match exactly
            val_f1 = metrics['val_f1'].item()
            
            current_epoch = trainer.current_epoch
            print(f"\nEpoch {current_epoch}: Validation F1 = {val_f1:.4f}")
            trainer.logger.experiment.add_scalar('F1/Validation', val_f1, current_epoch)
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='{epoch}-{val_f1:.4f}',
            monitor='val_f1',
            mode='max',
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_f1',
            patience=3,
            mode='max',
            min_delta=0.001
        ),
        F1LoggingCallback()
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision='16-mixed' if args.use_fp16 else '32'  # Updated precision format
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Save final model
    trainer.save_checkpoint(Path(args.checkpoint_dir) / "final_model.ckpt")

def main():
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_length', type=int, default=128)
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--run_name', type=str, default='emotion-detection')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    train(args)

if __name__ == "__main__":
    main()