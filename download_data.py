import os
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import emoji
from typing import List, Dict, Optional, Union
import shutil
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import login, hf_hub_download

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)

class DataProcessor:
    def __init__(self, base_dir: str = 'data'):
        self.base_dir = Path(base_dir)
        self.target_emotions = ['anger', 'fear', 'joy', 'sadness', 'surprise']
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for data storage with error handling."""
        dirs = [
            'semeval/train',
            'semeval/dev',
            'semeval/test',
            'goemotions',
            'emotions',
            'combined'
        ]
        for dir_path in dirs:
            try:
                (self.base_dir / dir_path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logging.error(f"Failed to create directory {dir_path}: {str(e)}")
                raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
        
        # Remove emojis
        text = emoji.replace_emoji(text, '')
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Basic cleaning
        text = text.strip()
        text = text.replace("[NAME]", "")
        text = text.replace("[USER]", "")
        text = text.replace("[URL]", "")
        
        return text

    def verify_emotion_columns(self, df: pd.DataFrame) -> bool:
        """Verify that DataFrame has required emotion columns."""
        return all(emotion in df.columns for emotion in self.target_emotions)

    def process_semeval_data(self):
        """Process SemEval data with validation only."""
        logging.info("Processing SemEval data...")
        
        files = {
            'train': 'dataset/semeval/track_a/train-eng.csv',
            'dev': 'dataset/semeval/track_a/dev-eng.csv',
            'test': 'dataset/semeval/track_a/test-eng.csv'
        }
        
        processed_data = {}
        
        for split, src_path in files.items():
            try:
                if not os.path.exists(src_path):
                    raise FileNotFoundError(f"SemEval file not found: {src_path}")
                
                # Just copy the file directly
                dest_path = self.base_dir / 'semeval' / split / 'eng.csv'
                shutil.copy2(src_path, dest_path)
                
                # Load for statistics only
                df = pd.read_csv(src_path)
                processed_data[split] = df
                
                # Verify columns for non-test data (but don't modify)
                if split != 'test':
                    if not self.verify_emotion_columns(df):
                        raise ValueError(f"Missing emotion columns in {src_path}")
                
                logging.info(f"Processed {split} split: {len(df)} samples")
                
            except Exception as e:
                logging.error(f"Error processing {split} split: {str(e)}")
                raise
        
        return processed_data

    def process_goemotions(self, max_workers: int = 4):
        """Process GoEmotions dataset with parallel processing."""
        logging.info("Processing GoEmotions dataset...")
        
        # Emotion mapping from GoEmotions to target emotions (using the exact same mapping)
        EMOTION_MAPPING = {
            'anger': [2, 3],           # anger, annoyance
            'fear': [14, 19],          # fear, nervousness
            'joy': [13, 17, 18, 20],  # excitement, joy, love, optimism
            'sadness': [9, 16, 24, 25], # disappointment, grief, remorse, sadness
            'surprise': [22, 26]        # realization, surprise
        }
        
        # Label map for debugging
        LABEL_MAP = {
            0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 
            4: 'approval', 5: 'caring', 6: 'confusion', 7: 'curiosity', 
            8: 'desire', 9: 'disappointment', 10: 'disapproval', 11: 'disgust',
            12: 'embarrassment', 13: 'excitement', 14: 'fear', 15: 'gratitude',
            16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness', 
            20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief',
            24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'
        }
        
        try:
            # Load dataset exactly as before
            dataset = load_dataset("google-research-datasets/go_emotions", 'simplified')
            logging.info("GoEmotions dataset loaded successfully")
            
            # Process each split
            split_mapping = {
                'train': 'train',
                'validation': 'val',
                'test': 'test'
            }
            
            results = {}
            for original_split, new_split in split_mapping.items():
                split_data = dataset[original_split]
                logging.info(f"Processing {original_split} split: {len(split_data)} examples")
                
                # Show sample mappings for debugging
                if original_split == 'train':
                    logging.info("\nSample of original labels (first 5 examples):")
                    for i in range(min(5, len(split_data))):
                        text = split_data[i]['text']
                        original_labels = [LABEL_MAP[idx] for idx in split_data[i]['labels']]
                        logging.info(f"Text: {text}")
                        logging.info(f"Original emotions: {original_labels}")
                
                # Create multi-hot encoded columns
                emotion_columns = {emotion: [] for emotion in self.target_emotions}
                cleaned_texts = []
                
                # Process all examples
                for example in tqdm(split_data, desc=f"Processing {original_split}"):
                    # Clean text
                    cleaned_text = self.clean_text(example['text'])
                    cleaned_texts.append(cleaned_text)
                    
                    # Convert labels
                    label_indices = set(example['labels'])
                    for emotion, indices in EMOTION_MAPPING.items():
                        has_emotion = any(idx in label_indices for idx in indices)
                        emotion_columns[emotion].append(1 if has_emotion else 0)
                
                # Create DataFrame
                result_df = pd.DataFrame({
                    'text': cleaned_texts,
                    **emotion_columns
                })
                
                # Remove rows with no emotions
                has_emotion = result_df[self.target_emotions].sum(axis=1) > 0
                result_df = result_df[has_emotion]
                
                # Save processed file
                output_file = self.base_dir / 'goemotions' / f'{new_split}.csv'
                result_df.to_csv(output_file, index=False)
                results[new_split] = result_df
                
                # Print statistics
                logging.info(f"\nStatistics for {new_split} split:")
                logging.info(f"Total samples: {len(result_df)}")
                for emotion in self.target_emotions:
                    count = result_df[emotion].sum()
                    percentage = (count / len(result_df)) * 100
                    logging.info(f"{emotion}: {count} ({percentage:.2f}%)")
            
            return results
            
        except Exception as e:
            logging.error(f"Failed to process GoEmotions dataset: {str(e)}")
            raise

    def process_emotions_dataset(self):
        """Process DAIR-AI Emotions dataset."""
        logging.info("Processing DAIR-AI Emotions dataset...")
        
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset("dair-ai/emotion", 'split')
            logging.info("DAIR-AI Emotions dataset loaded successfully")
            
            # Label mapping from DAIR-AI (0-5) to our format
            label_mapping = {
                0: {'sadness': 1},    # sadness
                1: {'joy': 1},        # joy
                2: {'love': 1},       # love maps to joy in our case
                3: {'anger': 1},      # anger
                4: {'fear': 1},       # fear
                5: {'surprise': 1}    # surprise
            }
            
            # Process each split
            split_mapping = {
                'train': 'train',
                'validation': 'val',
                'test': 'test'
            }
            
            results = {}
            for original_split, new_split in split_mapping.items():
                split_data = dataset[original_split]
                logging.info(f"Processing {original_split} split: {len(split_data)} examples")
                
                processed_data = []
                for example in tqdm(split_data, desc=f"Processing {original_split}"):
                    # Clean text
                    cleaned_text = self.clean_text(example['text'])
                    if not cleaned_text:
                        continue
                    
                    # Initialize emotion labels
                    labels = {emotion: 0 for emotion in self.target_emotions}
                    
                    # Map the label
                    label_dict = label_mapping[example['label']]
                    labels.update(label_dict)
                    
                    processed_data.append({
                        'text': cleaned_text,
                        **labels
                    })
                
                # Create DataFrame and save
                df = pd.DataFrame(processed_data)
                output_file = self.base_dir / 'emotions' / f'{new_split}.csv'
                df.to_csv(output_file, index=False)
                results[new_split] = df
                
                # Print statistics
                logging.info(f"\nStatistics for {new_split} split:")
                logging.info(f"Total samples: {len(df)}")
                for emotion in self.target_emotions:
                    count = df[emotion].sum()
                    percentage = (count / len(df)) * 100
                    logging.info(f"{emotion}: {count} ({percentage:.2f}%)")
            
            return results
            
        except Exception as e:
            logging.error(f"Failed to process DAIR-AI Emotions dataset: {str(e)}")
            raise

    def combine_datasets(self, augment: bool = True):
        """Combine and balance datasets with optional augmentation."""
        logging.info("Combining datasets...")
        
        dfs = []
        
        try:
            # Load all training data
            semeval_train = pd.read_csv(self.base_dir / 'semeval/train/eng.csv')
            semeval_dev = pd.read_csv(self.base_dir / 'semeval/dev/eng.csv')
            goemotions = pd.read_csv(self.base_dir / 'goemotions/train.csv')
            emotions = pd.read_csv(self.base_dir / 'emotions/train.csv')
            
            # Add source information
            semeval_train['source'] = 'semeval_train'
            semeval_dev['source'] = 'semeval_dev'
            goemotions['source'] = 'goemotions'
            emotions['source'] = 'emotions'
            
            # Keep only necessary columns
            required_columns = ['text'] + self.target_emotions + ['source']
            
            semeval_train = semeval_train[required_columns]
            semeval_dev = semeval_dev[required_columns]
            goemotions = goemotions[required_columns]
            emotions = emotions[required_columns]
            
            dfs.extend([semeval_train, semeval_dev, goemotions, emotions])
            
        except Exception as e:
            logging.error(f"Error loading datasets: {str(e)}")
            raise
        
        # Combine all data
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Print distribution before balancing
        print("\nDistribution before balancing:")
        for emotion in self.target_emotions:
            counts = {}
            for source in ['semeval_train', 'semeval_dev', 'goemotions', 'emotions']:
                source_df = combined_df[combined_df['source'] == source]
                if len(source_df) > 0:
                    count = source_df[emotion].sum()
                    percentage = (count / len(source_df)) * 100
                    counts[source] = (count, percentage)
            
            print(f"\n{emotion.capitalize()}:")
            for source, (count, percentage) in counts.items():
                print(f"{source}: {count} ({percentage:.2f}%)")
        
        # Balance dataset
        if augment:
            combined_df = self._balance_dataset(combined_df)
            
            # Print distribution after balancing
            print("\nDistribution after balancing:")
            for emotion in self.target_emotions:
                count = combined_df[emotion].sum()
                percentage = (count / len(combined_df)) * 100
                print(f"{emotion}: {count} ({percentage:.2f}%)")
        
        # Create new train/val split from combined data
        train_df, val_df = train_test_split(
            combined_df, 
            test_size=0.1,
            stratify=combined_df[self.target_emotions].values,
            random_state=42
        )
        
        # Save final datasets
        train_df.to_csv(self.base_dir / 'combined/train.csv', index=False)
        val_df.to_csv(self.base_dir / 'combined/dev.csv', index=False)
        
        # Print final statistics
        print(f"\nFinal dataset sizes:")
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        
        return train_df, val_df

    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance dataset using various augmentation techniques."""
        balanced_data = []
        target_samples = max(df[emotion].sum() for emotion in self.target_emotions)
        
        for emotion in self.target_emotions:
            # Get samples with this emotion
            emotion_samples = df[df[emotion] == 1]
            current_samples = len(emotion_samples)
            
            if current_samples < target_samples:
                # Calculate how many samples to generate
                samples_to_generate = target_samples - current_samples
                
                # Generate synthetic samples
                synthetic_samples = self._generate_synthetic_samples(
                    emotion_samples,
                    samples_to_generate
                )
                
                balanced_data.append(synthetic_samples)
        
        # Combine original and synthetic data
        if balanced_data:
            return pd.concat([df] + balanced_data, ignore_index=True)
        return df

    def _generate_synthetic_samples(self, samples: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Generate synthetic samples using various augmentation techniques."""
        synthetic_samples = []
        
        for _ in range(n_samples):
            # Randomly select a sample
            sample = samples.sample(n=1).iloc[0]
            
            # Apply random augmentation
            aug_type = np.random.choice(['swap', 'delete', 'both'])
            text = sample['text'].split()
            
            if len(text) > 3:  # Only augment if text is long enough
                if aug_type in ['swap', 'both']:
                    # Swap random words
                    idx1, idx2 = np.random.choice(len(text), 2, replace=False)
                    text[idx1], text[idx2] = text[idx2], text[idx1]
                
                if aug_type in ['delete', 'both']:
                    # Delete random word
                    idx = np.random.randint(len(text))
                    text.pop(idx)
            
            # Create new sample with only necessary columns
            new_sample = {
                'text': ' '.join(text),
                'source': sample['source']
            }
            # Add emotion columns
            for emotion in self.target_emotions:
                new_sample[emotion] = sample[emotion]
            
            synthetic_samples.append(new_sample)
        
        return pd.DataFrame(synthetic_samples)

    def print_data_structure(self):
        """Print the current data directory structure and file status."""
        print("\nData Directory Structure:")
        print("------------------------")
        
        def print_tree(path: Path, prefix: str = ""):
            # Print current path
            if path.is_file():
                size = path.stat().st_size / (1024 * 1024)  # Size in MB
                print(f"{prefix}└── {path.name} ({size:.1f}MB)")
            else:
                print(f"{prefix}└── {path.name}/")
                
                # Recursively print children
                children = sorted(list(path.iterdir()))
                for child in children:
                    print_tree(child, prefix + "    ")
        
        print_tree(self.base_dir)
        
        print("\nRequired Input Files:")
        print("-------------------")
        required_files = {
            'SemEval Train': 'dataset/semeval/track_a/train-eng.csv',
            'SemEval Dev': 'dataset/semeval/track_a/dev-eng.csv',
            'SemEval Test': 'dataset/semeval/track_a/test-eng.csv'
        }
        
        for name, path in required_files.items():
            status = "✓ Found" if os.path.exists(path) else "✗ Missing"
            print(f"{name}: {status}")

def main():
    try:
        processor = DataProcessor()
        
        # Print initial structure
        processor.print_data_structure()
        
        # Process all datasets
        processor.process_semeval_data()
        processor.process_goemotions()
        processor.process_emotions_dataset()
        
        # Combine datasets with augmentation
        final_dataset = processor.combine_datasets(augment=True)
        
        # Print final structure
        print("\nFinal Data Structure:")
        processor.print_data_structure()
        
        logging.info("Data processing completed successfully")
        
    except Exception as e:
        logging.error(f"Data processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()