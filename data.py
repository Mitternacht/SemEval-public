import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import RobertaTokenizer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Dict

class EmotionDataset(Dataset):
    def __init__(
        self, 
        data_file: str, 
        tokenizer: RobertaTokenizer, 
        max_length: int = 128,
        augment: bool = False
    ):
        self.data = pd.read_csv(data_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
        
        # Check if this is a test set (no labels)
        self.is_test = all(self.data[label].isna().all() for label in self.emotion_labels)
        
        # Create label counts for weighted sampling
        if not self.is_test:
            self.label_counts = self._calculate_label_counts()
            self.sample_weights = self._calculate_sample_weights()

    def _calculate_label_counts(self) -> Dict[str, int]:
        """Calculate the count of each label in the dataset."""
        return {
            label: self.data[label].sum()
            for label in self.emotion_labels
        }

    def _calculate_sample_weights(self) -> torch.Tensor:
        """Calculate weights for each sample based on its labels."""
        weights = []
        total_samples = len(self.data)
        
        for idx in range(total_samples):
            sample_labels = [
                self.data.iloc[idx][label]
                for label in self.emotion_labels
            ]
            # Weight based on inverse frequency of present labels
            weight = sum(
                1.0 / self.label_counts[label] 
                for label, present in zip(self.emotion_labels, sample_labels) 
                if present
            )
            weights.append(weight if weight > 0 else 1.0)
        
        return torch.tensor(weights, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        
        # Handle test set differently
        if self.is_test:
            labels = torch.zeros(len(self.emotion_labels), dtype=torch.float32)
        else:
            labels = torch.tensor(
                [float(self.data.iloc[idx][label]) for label in self.emotion_labels], 
                dtype=torch.float32
            )
        
        # Basic tokenization
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze().long(),
            'attention_mask': encoding['attention_mask'].squeeze().long(),
            'labels': labels
        }

class EmotionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        test_file: str,
        batch_size: int = 8,
        max_length: int = 128,
        use_augmentation: bool = False,
        aug_prob: float = 0.3,
        num_workers: int = 4
    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.use_augmentation = use_augmentation
        self.aug_prob = aug_prob
        self.num_workers = num_workers
        
        # Add special tokens for emotions
        special_tokens = ['[ANGER]', '[FEAR]', '[JOY]', '[SADNESS]', '[SURPRISE]']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = EmotionDataset(
                self.train_file, 
                self.tokenizer, 
                self.max_length,
                augment=self.use_augmentation
            )
            self.val_dataset = EmotionDataset(
                self.val_file, 
                self.tokenizer, 
                self.max_length
            )
            
            # Calculate class weights for loss function
            train_data = pd.read_csv(self.train_file)
            self.class_weights = self._calculate_class_weights(train_data)

        if stage == 'test' or stage is None:
            self.test_dataset = EmotionDataset(
                self.test_file, 
                self.tokenizer, 
                self.max_length
            )

    def _calculate_class_weights(self, data):
        """Calculate balanced class weights."""
        emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
        pos_weights = []
        
        for label in emotion_labels:
            neg_count = len(data[data[label] == 0])
            pos_count = len(data[data[label] == 1])
            # Square root scaling for softer weighting
            weight = (neg_count / pos_count if pos_count > 0 else 1.0) ** 0.5
            pos_weights.append(weight)
        
        return torch.tensor(pos_weights, dtype=torch.float)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True  # Prevent issues with batch norm
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def print_dataset_stats(self):
        """Print and visualize detailed dataset statistics."""
        datasets = {
            'Train': self.train_file,
            'Validation': self.val_file,
            'Test': self.test_file
        }
        
        stats = {}
        plt.figure(figsize=(20, 15))
        
        for idx, (name, file) in enumerate(datasets.items()):
            data = pd.read_csv(file)
            emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
            
            # Basic statistics
            total_samples = len(data)
            label_counts = {label: data[label].sum() for label in emotion_labels}
            label_percentages = {
                label: (count/total_samples)*100 
                for label, count in label_counts.items()
            }
            
            # Multi-label distribution
            label_combinations = data[emotion_labels].apply(
                lambda x: tuple(x), axis=1
            )
            combination_counts = Counter(label_combinations)
            
            # Co-occurrence matrix
            cooc_matrix = np.zeros((len(emotion_labels), len(emotion_labels)))
            for i, em1 in enumerate(emotion_labels):
                for j, em2 in enumerate(emotion_labels):
                    cooc_matrix[i, j] = ((data[em1] == 1) & (data[em2] == 1)).sum()
            
            # Store statistics
            stats[name] = {
                'total_samples': total_samples,
                'label_counts': label_counts,
                'label_percentages': label_percentages,
                'avg_labels_per_sample': data[emotion_labels].sum(axis=1).mean(),
                'multi_label_dist': {
                    f'{sum(combo)}_labels': count 
                    for combo, count in combination_counts.items()
                },
                'cooc_matrix': cooc_matrix
            }
            
            # Plot distribution
            plt.subplot(3, 3, idx*3 + 1)
            sns.barplot(
                x=list(label_percentages.keys()),
                y=list(label_percentages.values())
            )
            plt.title(f'{name} Set Distribution')
            plt.xticks(rotation=45)
            plt.ylabel('Percentage')
            
            # Plot label count distribution
            plt.subplot(3, 3, idx*3 + 2)
            labels_per_sample = data[emotion_labels].sum(axis=1)
            sns.histplot(labels_per_sample, bins=range(7))
            plt.title(f'{name} Labels per Sample')
            plt.xlabel('Number of Labels')
            plt.ylabel('Count')
            
            # Plot co-occurrence matrix
            plt.subplot(3, 3, idx*3 + 3)
            sns.heatmap(
                cooc_matrix,
                annot=True,
                fmt='.0f',
                xticklabels=emotion_labels,
                yticklabels=emotion_labels
            )
            plt.title(f'{name} Co-occurrence Matrix')
        
        plt.tight_layout()
        plt.savefig('plots/data_distribution.png')
        plt.close()
        
        # Print statistics
        for name, stat in stats.items():
            print(f"\n{name} Set Statistics:")
            print(f"Total samples: {stat['total_samples']}")
            print("\nLabel distribution:")
            for label, count in stat['label_counts'].items():
                print(f"{label}: {count} ({stat['label_percentages'][label]:.2f}%)")
            print(f"\nAverage labels per sample: {stat['avg_labels_per_sample']:.2f}")
            print("\nMulti-label distribution:")
            for n_labels, count in stat['multi_label_dist'].items():
                print(f"{n_labels}: {count}")