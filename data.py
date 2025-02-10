import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import RobertaTokenizer
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import logging
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class EmotionDataset(Dataset):
    def __init__(
        self, 
        data_file: str, 
        tokenizer: RobertaTokenizer, 
        max_length: int = 128,
        augment: bool = False,
        cache_dir: Optional[str] = None
    ):
        self.data_file = Path(data_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load and preprocess data
        self.data = self._load_data()
        
        # Check if this is a test set (no labels)
        self.is_test = self._check_if_test()
        
        # Create label information
        if not self.is_test:
            self.label_info = self._calculate_label_info()
            self.sample_weights = self._calculate_sample_weights()
            
        # Cache tokenized data if cache directory is provided
        if self.cache_dir:
            self._cache_tokenized_data()

    def _load_data(self) -> pd.DataFrame:
        """Load and validate data."""
        try:
            df = pd.read_csv(self.data_file)
            
            # Validate required columns
            required_cols = ['text'] + (
                self.emotion_labels if not self._check_if_test(df) else []
            )
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading data from {self.data_file}: {str(e)}")
            raise

    def _check_if_test(self, df: Optional[pd.DataFrame] = None) -> bool:
        """Check if dataset is a test set."""
        df = df if df is not None else self.data
        return all(df[label].isna().all() for label in self.emotion_labels)

    def _calculate_label_info(self) -> Dict[str, Any]:
        """Calculate detailed label information."""
        info = {
            'counts': {},
            'frequencies': {},
            'co_occurrences': np.zeros((len(self.emotion_labels), len(self.emotion_labels))),
            'label_combinations': Counter()
        }
        
        # Calculate basic counts and frequencies
        for label in self.emotion_labels:
            count = self.data[label].sum()
            info['counts'][label] = count
            info['frequencies'][label] = count / len(self.data)
        
        # Calculate co-occurrences
        for i, label1 in enumerate(self.emotion_labels):
            for j, label2 in enumerate(self.emotion_labels):
                co_occur = ((self.data[label1] == 1) & (self.data[label2] == 1)).sum()
                info['co_occurrences'][i, j] = co_occur
        
        # Calculate label combinations
        label_combinations = self.data[self.emotion_labels].apply(
            lambda x: tuple(x), axis=1
        )
        info['label_combinations'].update(label_combinations)
        
        return info

    def _calculate_sample_weights(self) -> torch.Tensor:
        """Calculate weights for balanced sampling."""
        weights = []
        total_samples = len(self.data)
        
        for idx in range(total_samples):
            sample_labels = [
                self.data.iloc[idx][label]
                for label in self.emotion_labels
            ]
            # Weight based on inverse frequency of present labels
            weight = sum(
                1.0 / self.label_info['counts'][label] 
                for label, present in zip(self.emotion_labels, sample_labels) 
                if present
            )
            weights.append(weight if weight > 0 else 1.0)
        
        return torch.tensor(weights, dtype=torch.float)

    def _cache_tokenized_data(self):
        """Cache tokenized data to disk."""
        cache_file = self.cache_dir / f"{self.data_file.stem}_tokenized.pt"
        
        if not cache_file.exists():
            tokenized_data = []
            for idx in range(len(self.data)):
                encoding = self._tokenize_text(str(self.data.iloc[idx]['text']))
                tokenized_data.append(encoding)
            
            # Save to cache
            torch.save(tokenized_data, cache_file)
            self.tokenized_cache = tokenized_data
        else:
            self.tokenized_cache = torch.load(cache_file)

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text with error handling."""
        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            }
            
        except Exception as e:
            logging.error(f"Error tokenizing text: {text}\nError: {str(e)}")
            # Return empty tensors as fallback
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
            }

    def _augment_text(self, text: str) -> str:
        """Apply text augmentation techniques."""
        words = text.split()
        if len(words) <= 4:
            return text
            
        augmentations = []
        
        # Random word dropout
        if torch.rand(1) < 0.3:
            dropout_idx = torch.randint(0, len(words), (len(words)//10,))
            aug_words = [w for i, w in enumerate(words) if i not in dropout_idx]
            augmentations.append(' '.join(aug_words))
        
        # Random word swap
        if torch.rand(1) < 0.3:
            aug_words = words.copy()
            idx1, idx2 = torch.randint(0, len(words), (2,))
            aug_words[idx1], aug_words[idx2] = aug_words[idx2], aug_words[idx1]
            augmentations.append(' '.join(aug_words))
        
        # Return augmented text or original if no augmentations
        return augmentations[-1] if augmentations else text

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            # Get text and apply augmentation if needed
            text = str(self.data.iloc[idx]['text'])
            if self.augment and not self.is_test:
                text = self._augment_text(text)
            
            # Get tokenized data from cache if available
            if hasattr(self, 'tokenized_cache'):
                encoding = self.tokenized_cache[idx]
            else:
                encoding = self._tokenize_text(text)
            
            # Prepare labels
            if self.is_test:
                labels = torch.zeros(len(self.emotion_labels), dtype=torch.float32)
            else:
                labels = torch.tensor(
                    [float(self.data.iloc[idx][label]) for label in self.emotion_labels],
                    dtype=torch.float32
                )
            
            return {
                'input_ids': encoding['input_ids'].long(),
                'attention_mask': encoding['attention_mask'].long(),
                'labels': labels
            }
            
        except Exception as e:
            logging.error(f"Error getting item {idx}: {str(e)}")
            # Return empty tensors as fallback
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.zeros(len(self.emotion_labels), dtype=torch.float32)
            }

class EmotionDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for emotion detection"""
    
    # Class variables for label columns
    EMOTION_LABELS = ['anger', 'fear', 'joy', 'sadness', 'surprise']
    
    def __init__(self,
                 data_path: str,  # Changed from data_dir to data_path
                 batch_size: int = 32,
                 max_length: int = 128,
                 num_workers: int = 4):
        """
        Initialize EmotionDataModule
        
        Args:
            data_path: Direct path to CSV file
            batch_size: Batch size for dataloaders
            max_length: Maximum sequence length for tokenizer
            num_workers: Number of workers for dataloaders
        """
        super().__init__()
        self.data_path = Path(data_path)  # Direct path to CSV file
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        
        # Initialize RobertaTokenizer with emotion-specific tokens
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        special_tokens = ['[ANGER]', '[FEAR]', '[JOY]', '[SADNESS]', '[SURPRISE]']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    def setup(self, stage: Optional[str] = None):
        """Load and prepare data for given stage"""
        if stage == 'fit' or stage is None:
            # For training, we need both train and validation datasets
            if 'train.csv' in str(self.data_path):
                # When loading training data, also load validation data
                self.train_dataset = EmotionDataset(
                    self.data_path,  # train.csv
                    self.tokenizer,
                    max_length=self.max_length,
                    augment=True  # Enable augmentation for training
                )
                
                # Load validation dataset
                val_path = Path(self.data_path).parent / 'dev.csv'
                self.val_dataset = EmotionDataset(
                    val_path,
                    self.tokenizer,
                    max_length=self.max_length,
                    augment=False  # No augmentation for validation
                )
                
                logging.info(f"Loaded {len(self.train_dataset)} training samples")
                logging.info(f"Loaded {len(self.val_dataset)} validation samples")
            else:
                # For evaluation only, use the dataset as is
                self.val_dataset = EmotionDataset(
                    self.data_path,
                    self.tokenizer,
                    max_length=self.max_length,
                    augment=False
                )
                logging.info(f"Loaded {len(self.val_dataset)} evaluation samples")

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def _create_dataloader(self, dataset: EmotionDataset, shuffle: bool) -> DataLoader:
        """Create DataLoader with error handling and optimal settings."""
        try:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle and not hasattr(dataset, 'sampler'),
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False,
                drop_last=shuffle,  # Only drop last batch during training
                prefetch_factor=2 if self.num_workers > 0 else None,
                worker_init_fn=lambda worker_id: np.random.seed(worker_id)
            )
        except Exception as e:
            logging.error(f"Error creating DataLoader: {str(e)}")
            raise

    def print_dataset_stats(self):
        """Print and visualize detailed dataset statistics."""
        if not hasattr(self, 'train_dataset'):
            logging.warning("Dataset statistics not available. Call setup() first.")
            return
            
        try:
            self._plot_dataset_statistics()
        except Exception as e:
            logging.error(f"Error plotting dataset statistics: {str(e)}")