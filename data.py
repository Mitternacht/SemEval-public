import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import RobertaTokenizer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class EmotionDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128):
        self.data = pd.read_csv(data_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
        # Check if this is a test set (no labels)
        self.is_test = all(self.data[label].isna().all() for label in self.emotion_labels)

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
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }

class EmotionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        test_file: str,
        batch_size: int = 16,
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

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = EmotionDataset(
                self.train_file, 
                self.tokenizer, 
                self.max_length
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
        emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
        pos_weights = []
        for label in emotion_labels:
            neg_count = len(data[data[label] == 0])
            pos_count = len(data[data[label] == 1])
            pos_weights.append(
                (neg_count / pos_count if pos_count > 0 else 1.0) ** 0.5
            )
        return torch.tensor(pos_weights, dtype=torch.float)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
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
        """Print and visualize dataset statistics."""
        datasets = {
            'Train': self.train_file,
            'Validation': self.val_file,
            'Test': self.test_file
        }
        
        stats = {}
        plt.figure(figsize=(15, 10))
        
        for idx, (name, file) in enumerate(datasets.items()):
            data = pd.read_csv(file)
            emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
            
            # Calculate statistics
            total_samples = len(data)
            label_counts = {label: data[label].sum() for label in emotion_labels}
            label_percentages = {label: (count/total_samples)*100 
                               for label, count in label_counts.items()}
            
            # Calculate multi-label distribution
            label_combinations = data[emotion_labels].apply(
                lambda x: tuple(x), axis=1
            )
            combination_counts = Counter(label_combinations)
            
            # Store stats
            stats[name] = {
                'total_samples': total_samples,
                'label_counts': label_counts,
                'label_percentages': label_percentages,
                'avg_labels_per_sample': data[emotion_labels].sum(axis=1).mean(),
                'multi_label_dist': {
                    f'{sum(combo)}_labels': count 
                    for combo, count in combination_counts.items()
                }
            }
            
            # Plot distribution
            plt.subplot(1, 3, idx+1)
            sns.barplot(
                x=list(label_percentages.keys()),
                y=list(label_percentages.values())
            )
            plt.title(f'{name} Set Distribution')
            plt.xticks(rotation=45)
            plt.ylabel('Percentage')
        
        plt.tight_layout()
        plt.savefig('data_distribution.png')
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
