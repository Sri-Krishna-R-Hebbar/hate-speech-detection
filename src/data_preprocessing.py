import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class DataPreprocessor:
    def __init__(self, max_words=10000, max_len=100):
        """
        Initialize the data preprocessor
        
        Args:
            max_words: Maximum number of words to keep in vocabulary
            max_len: Maximum length of sequences
        """
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = None
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove RT (retweet)
        text = re.sub(r'\brt\b', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        text = ' '.join([word for word in text.split() if word not in self.stop_words])
        
        return text
    
    def load_and_merge_datasets(self, racist_path, sexist_path):
        """
        Load and merge racist and sexist datasets
        
        Args:
            racist_path: Path to racist dataset CSV
            sexist_path: Path to sexist dataset CSV
            
        Returns:
            Merged dataframe with unified labels
        """
        print("Loading datasets...")
        
        # Load datasets
        df_racist = pd.read_csv(racist_path)
        df_sexist = pd.read_csv(sexist_path)
        
        print(f"Racist dataset shape: {df_racist.shape}")
        print(f"Sexist dataset shape: {df_sexist.shape}")
        
        # Print actual column names for debugging
        print(f"Racist dataset columns: {df_racist.columns.tolist()}")
        print(f"Sexist dataset columns: {df_sexist.columns.tolist()}")
        
        # Standardize column names - handle different possible formats
        # Expected columns: 'Text' and 'Annotation_oh_label' OR 'label'
        column_mapping = {}
        
        # Handle text column
        for col in df_racist.columns:
            if col.lower() == 'text' or col == 'Text':
                column_mapping[col] = 'text'
        
        # Handle label column
        for col in df_racist.columns:
            if 'label' in col.lower() or col == 'Annotation_oh_label':
                column_mapping[col] = 'label'
        
        # Rename columns in both dataframes
        df_racist = df_racist.rename(columns=column_mapping)
        
        # Same for sexist dataset
        column_mapping = {}
        for col in df_sexist.columns:
            if col.lower() == 'text' or col == 'Text':
                column_mapping[col] = 'text'
        for col in df_sexist.columns:
            if 'label' in col.lower() or col == 'Annotation_oh_label':
                column_mapping[col] = 'label'
        
        df_sexist = df_sexist.rename(columns=column_mapping)
        
        # Verify we have required columns
        if 'text' not in df_racist.columns or 'label' not in df_racist.columns:
            raise ValueError(f"Racist dataset missing required columns. Has: {df_racist.columns.tolist()}")
        if 'text' not in df_sexist.columns or 'label' not in df_sexist.columns:
            raise ValueError(f"Sexist dataset missing required columns. Has: {df_sexist.columns.tolist()}")
        
        # Map labels: 0 (none) stays 0, 1 (racist) becomes 1, 1 (sexist) becomes 2
        df_racist['label'] = df_racist['label'].map({0: 0, 1: 1})  # none: 0, racist: 1
        df_sexist['label'] = df_sexist['label'].map({0: 0, 1: 2})  # none: 0, sexist: 2
        
        # Combine datasets
        df_combined = pd.concat([df_racist, df_sexist], ignore_index=True)
        
        # Remove any rows with missing text or labels
        df_combined = df_combined.dropna(subset=['text', 'label'])
        
        print(f"\nCombined dataset shape: {df_combined.shape}")
        print(f"Label distribution:\n{df_combined['label'].value_counts().sort_index()}")
        
        return df_combined
    
    def preprocess_data(self, df):
        """
        Preprocess the dataframe
        
        Args:
            df: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        print("\nCleaning text data...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Remove empty texts after cleaning
        df = df[df['cleaned_text'].str.len() > 0]
        
        print(f"Dataset shape after cleaning: {df.shape}")
        
        return df
    
    def tokenize_and_pad(self, texts, fit=True):
        """
        Tokenize and pad text sequences
        
        Args:
            texts: List of text strings
            fit: Whether to fit the tokenizer (True for training data)
            
        Returns:
            Padded sequences
        """
        if fit:
            self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts)
            print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        return padded
    
    def prepare_data(self, racist_path, sexist_path, test_size=0.2, val_size=0.1, random_state=42):
        """
        Complete data preparation pipeline
        
        Args:
            racist_path: Path to racist dataset
            sexist_path: Path to sexist dataset
            test_size: Proportion of test data
            val_size: Proportion of validation data (from training data)
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Load and merge datasets
        df = self.load_and_merge_datasets(racist_path, sexist_path)
        
        # Preprocess
        df = self.preprocess_data(df)
        
        # Split data
        X = df['cleaned_text'].values
        y = df['label'].values
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Tokenize and pad
        print("\nTokenizing and padding sequences...")
        X_train_padded = self.tokenize_and_pad(X_train, fit=True)
        X_val_padded = self.tokenize_and_pad(X_val, fit=False)
        X_test_padded = self.tokenize_and_pad(X_test, fit=False)
        
        return X_train_padded, X_val_padded, X_test_padded, y_train, y_val, y_test
    
    def get_vocab_size(self):
        """Get the vocabulary size"""
        if self.tokenizer is None:
            return 0
        return min(len(self.tokenizer.word_index) + 1, self.max_words)