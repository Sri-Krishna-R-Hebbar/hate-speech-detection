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
        
        # MINIMAL stopword removal - only remove truly useless words
        # Keep ALL words that could be important for hate speech detection
        all_stopwords = set(stopwords.words('english'))
        
        # Words to DEFINITELY KEEP (critical for hate speech detection)
        keep_words = {
            # NEGATIONS - ABSOLUTELY CRITICAL!
            'not', 'no', 'nor', 'neither', 'never', 'none', 'nothing', 'nowhere',
            'nobody', 'noone', 
            
            # Negation contractions
            'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
            'isn', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
            'won', 'wouldn', 'don', 'aren\'t', 'can\'t', 'couldn\'t', 'didn\'t',
            'doesn\'t', 'don\'t', 'hadn\'t', 'hasn\'t', 'haven\'t', 'isn\'t',
            'mustn\'t', 'needn\'t', 'shan\'t', 'shouldn\'t', 'wasn\'t', 'weren\'t',
            'won\'t', 'wouldn\'t',
            
            # QUANTIFIERS - Show intensity/scope
            'all', 'every', 'any', 'some', 'most', 'few', 'more', 'less', 'least',
            'much', 'many', 'only', 'just', 'too', 'very', 'so', 'enough', 'such',
            'same', 'other', 'another', 'both', 'each', 'either',
            
            # PRONOUNS - Identify targets
            'she', 'he', 'her', 'him', 'his', 'hers', 'they', 'them', 'their',
            'theirs', 'who', 'whom', 'whose', 'which', 'what', 'whoever',
            'whomever', 'you', 'your', 'yours', 'we', 'our', 'ours',
            
            # MODAL VERBS - Show assertions/requirements
            'should', 'would', 'could', 'can', 'will', 'must', 'might', 'may',
            'shall', 'ought',
            
            # PREPOSITIONS - Show relationships
            'from', 'to', 'for', 'with', 'against', 'about', 'like', 'as',
            'at', 'by', 'in', 'on', 'of', 'off', 'over', 'under', 'between',
            'among', 'through', 'during', 'before', 'after', 'above', 'below',
            
            # CONJUNCTIONS - Connect ideas
            'and', 'or', 'but', 'because', 'if', 'when', 'where', 'while',
            'than', 'since', 'unless', 'until', 'whether', 'though', 'although',
            
            # VERBS - Actions/states
            'are', 'is', 'was', 'were', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'having', 'doing', 'am',
            
            # ADVERBS - Modify meaning
            'how', 'why', 'here', 'there', 'now', 'then', 'always', 'never',
            'often', 'sometimes', 'usually', 'really', 'quite', 'rather',
            
            # DETERMINERS
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            
            # OTHER IMPORTANT
            'as', 'well', 'back', 'even', 'still', 'also', 'own', 'same',
            'different', 'new', 'old', 'good', 'bad', 'first', 'last'
        }
        
        # Only remove TRULY useless stopwords (articles, etc that don't add meaning)
        useless_stopwords = {
            'i', 'me', 'my', 'myself', 'ours', 'ourselves', 'yourselves',
            'himself', 'herself', 'itself', 'themselves', 
            'again', 'further', 'once', 'down', 'out', 'up',
        }
        
        # Final stopwords: only the truly useless ones
        self.stop_words = useless_stopwords
        
        print(f"[DEBUG] Using MINIMAL stopword removal: {len(self.stop_words)} words")
        print(f"[DEBUG] Keeping {len(keep_words)} critical words")
        
    def clean_text(self, text):
        """Clean and preprocess text data while preserving ALL important context"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtag symbols but keep the word
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove RT (retweet)
        text = re.sub(r'\brt\b', '', text)
        
        # CRITICAL: Expand contractions to preserve negations
        contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot",
            "can't've": "cannot have", "could've": "could have",
            "couldn't": "could not", "couldn't've": "could not have",
            "didn't": "did not", "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hadn't've": "had not have",
            "hasn't": "has not", "haven't": "have not",
            "he'd": "he would", "he'd've": "he would have",
            "he'll": "he will", "he'll've": "he will have",
            "he's": "he is", "how'd": "how did",
            "how'd'y": "how do you", "how'll": "how will",
            "how's": "how is", "i'd": "i would",
            "i'd've": "i would have", "i'll": "i will",
            "i'll've": "i will have", "i'm": "i am",
            "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'd've": "it would have",
            "it'll": "it will", "it'll've": "it will have",
            "it's": "it is", "let's": "let us",
            "ma'am": "madam", "mayn't": "may not",
            "might've": "might have", "mightn't": "might not",
            "mightn't've": "might not have", "must've": "must have",
            "mustn't": "must not", "mustn't've": "must not have",
            "needn't": "need not", "needn't've": "need not have",
            "o'clock": "of the clock", "oughtn't": "ought not",
            "oughtn't've": "ought not have", "shan't": "shall not",
            "sha'n't": "shall not", "shan't've": "shall not have",
            "she'd": "she would", "she'd've": "she would have",
            "she'll": "she will", "she'll've": "she will have",
            "she's": "she is", "should've": "should have",
            "shouldn't": "should not", "shouldn't've": "should not have",
            "so've": "so have", "so's": "so is",
            "that'd": "that would", "that'd've": "that would have",
            "that's": "that is", "there'd": "there would",
            "there'd've": "there would have", "there's": "there is",
            "they'd": "they would", "they'd've": "they would have",
            "they'll": "they will", "they'll've": "they will have",
            "they're": "they are", "they've": "they have",
            "to've": "to have", "wasn't": "was not",
            "we'd": "we would", "we'd've": "we would have",
            "we'll": "we will", "we'll've": "we will have",
            "we're": "we are", "we've": "we have",
            "weren't": "were not", "what'll": "what will",
            "what'll've": "what will have", "what're": "what are",
            "what's": "what is", "what've": "what have",
            "when's": "when is", "when've": "when have",
            "where'd": "where did", "where's": "where is",
            "where've": "where have", "who'll": "who will",
            "who'll've": "who will have", "who's": "who is",
            "who've": "who have", "why's": "why is",
            "why've": "why have", "will've": "will have",
            "won't": "will not", "won't've": "will not have",
            "would've": "would have", "wouldn't": "would not",
            "wouldn't've": "would not have", "y'all": "you all",
            "y'all'd": "you all would", "y'all'd've": "you all would have",
            "y'all're": "you all are", "y'all've": "you all have",
            "you'd": "you would", "you'd've": "you would have",
            "you'll": "you will", "you'll've": "you will have",
            "you're": "you are", "you've": "you have"
        }
        
        # Apply contractions
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove special characters and numbers but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # MINIMAL stopword removal - only remove truly useless words
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        text = ' '.join(filtered_words)
        
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
        
        # Standardize column names
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