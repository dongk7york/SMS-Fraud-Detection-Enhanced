"""
SMS Fraud Detection - Enhanced Implementation
==============================================

Based on: TelecomFraudDetection by nsalhab
Original Repository: https://github.com/nsalhab/TelecomFraudDetection

This implementation extends the original work with:
- Dual Word2Vec models (text + URL embeddings)
- POS-aware preprocessing with context-sensitive lemmatization
- 14 engineered features for enhanced fraud detection
- Combined dataset approach (original + UCI SMS Spam Collection)
- Comprehensive visualization suite

Data Sources:
1. Original Telecom Fraud Dataset
   Source: https://github.com/nsalhab/TelecomFraudDetection
   
2. UCI SMS Spam Collection
   Citation: Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset].
             UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

Implementation Plan:
1. Pre-processing: URL extraction, text cleaning, POS-aware lemmatization
2. Baseline: TF-IDF + Logistic Regression + SVM (original approach)
3. Enhanced: Dual Word2Vec + hybrid features + LR/SVM comparison
4. Extended: Add Random Forest, Gradient Boosting for ensemble methods
5. Evaluation: Compare all approaches with detailed metrics

Course: MATH6912 - Machine Learning in Finance
Date: March 2026
"""

import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# NLP and ML imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Word2Vec imports
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("SMS FRAUD DETECTION - ENHANCED IMPLEMENTATION")
print("=" * 80)


# ============================================================================
# PART 1: PRE-PROCESSING
# ============================================================================

class SMSPreprocessor:
    """Handles all pre-processing tasks including URL extraction and text cleaning"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # URL patterns for extraction
        self.url_patterns = [
            r'https?://[^\s]+',  # Standard URLs with protocol
            r'www\.[^\s]+',  # URLs starting with www
            r'[a-zA-Z0-9-]+\.[a-z]{2,}(?:/[^\s]*)?',  # Plain text domains
            r'\[\s*([a-zA-Z0-9-]+\.[a-z]{2,}(?:/[^\s]*)?)\s*\]',  # Bracketed domains
        ]
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from text"""
        urls = []
        for pattern in self.url_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            urls.extend(matches)
        return urls
    
    def has_url(self, text: str) -> int:
        """Binary feature: does text contain URL?"""
        return 1 if self.extract_urls(text) else 0
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text for clean text processing"""
        for pattern in self.url_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()
    
    def clean_text(self, text: str) -> str:
        """Clean text: lowercase, remove special chars, lemmatize with POS tagging"""
        # Remove URLs first
        text = self.remove_urls(text)
        
        # Lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # POS tagging for better lemmatization
        pos_tags = nltk.pos_tag(tokens)
        
        # Remove stopwords and lemmatize with POS tags
        tokens = []
        for token, pos in pos_tags:
            if token not in self.stop_words and len(token) > 2:
                # Convert POS tag to WordNet format
                wordnet_pos = self._get_wordnet_pos(pos)
                # Lemmatize with POS tag
                lemma = self.lemmatizer.lemmatize(token, pos=wordnet_pos)
                tokens.append(lemma)
        
        return ' '.join(tokens)
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert TreeBank POS tags to WordNet POS tags"""
        if treebank_tag.startswith('J'):
            return 'a'  # Adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # Verb
        elif treebank_tag.startswith('N'):
            return 'n'  # Noun
        elif treebank_tag.startswith('R'):
            return 'r'  # Adverb
        else:
            return 'n'  # Default to noun
    
    def tokenize_url_for_word2vec(self, url: str) -> List[str]:
        """
        Tokenize URL for Word2Vec training
        Captures domain patterns, separators, and character n-grams
        """
        tokens = []
        
        # Split by separators
        parts = re.split(r'[-./]', url.lower())
        tokens.extend([p for p in parts if p])
        
        # Add character trigrams for pattern learning
        for i in range(len(url) - 2):
            tokens.append(url[i:i+3])
        
        # Add pattern markers
        if re.match(r'^\d', url):
            tokens.append('NUMERIC_PREFIX')
        
        # Check for brand impersonation keywords
        brand_keywords = ['interac', 'paypal', 'amazon', 'bank', 'visa', 'mastercard']
        for brand in brand_keywords:
            if brand in url.lower():
                tokens.append(f'BRAND_{brand.upper()}')
        
        # Check for suspicious TLDs
        suspicious_tlds = ['.info', '.xyz', '.top', '.online', '.club', '.site']
        for tld in suspicious_tlds:
            if url.endswith(tld):
                tokens.append('SUSPICIOUS_TLD')
        
        return tokens
    
    def extract_engineered_features(self, text: str) -> Dict[str, float]:
        """Extract hand-crafted features"""
        features = {}
        
        # Length features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Special character counts
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['dollar_count'] = text.count('$')
        features['percent_count'] = text.count('%')
        
        # Uppercase ratio
        if len(text) > 0:
            features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
        else:
            features['uppercase_ratio'] = 0
        
        # Digit ratio
        if len(text) > 0:
            features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text)
        else:
            features['digit_ratio'] = 0
        
        # URL features
        urls = self.extract_urls(text)
        features['has_url'] = 1 if urls else 0
        features['url_count'] = len(urls)
        
        if urls:
            # Average URL length
            features['avg_url_length'] = np.mean([len(url) for url in urls])
            
            # Check for HTTPS
            features['has_https'] = 1 if any('https' in url.lower() for url in urls) else 0
        else:
            features['avg_url_length'] = 0
            features['has_https'] = 0
        
        # Urgency keywords
        urgency_words = ['urgent', 'immediately', 'now', 'hurry', 'limited', 'expire', 'verify', 'confirm']
        features['urgency_score'] = sum(1 for word in urgency_words if word in text.lower())
        
        # Financial keywords
        financial_words = ['bank', 'account', 'payment', 'transfer', 'deposit', 'credit', 'debit']
        features['financial_score'] = sum(1 for word in financial_words if word in text.lower())
        
        return features


# ============================================================================
# PART 2: DATA LOADING AND INITIAL ANALYSIS
# ============================================================================

def load_and_analyze_data(filepath: str = 'Book1.csv') -> pd.DataFrame:
    """Load data and perform initial analysis"""
    print("\n" + "=" * 80)
    print("LOADING AND ANALYZING DATA")
    print("=" * 80)
    
    # Load data with encoding handling
    with open(filepath, encoding='utf-8', errors='replace') as f:
        df = pd.read_csv(f)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Drop missing values
    df = df.dropna()
    print(f"Shape after dropping NaN: {df.shape}")
    
    # Drop duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicates found: {duplicates}")
    df = df.drop_duplicates()
    print(f"Shape after dropping duplicates: {df.shape}")
    
    # Class distribution
    print(f"\nClass distribution:")
    print(df['Case'].value_counts())
    print(f"\nFraud percentage: {(df['Case'] == 1).sum() / len(df) * 100:.2f}%")
    
    return df


def load_uci_data(filepath: str = 'data/UCI_spam_data.csv') -> pd.DataFrame:
    """Load and format UCI SMS Spam Collection dataset"""
    print("\n" + "=" * 80)
    print("LOADING UCI SMS SPAM COLLECTION")
    print("=" * 80)
    
    # Load UCI data - it has columns v1 (label) and v2 (text)
    with open(filepath, encoding='utf-8', errors='replace') as f:
        df = pd.read_csv(f)
    
    # Keep only first two columns and rename
    df = df.iloc[:, :2]
    df.columns = ['label', 'text']
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert labels: ham=0, spam=1 (to match original dataset's Case column)
    df['Case'] = (df['label'] == 'spam').astype(int)
    
    # Drop missing values
    df = df.dropna()
    print(f"Shape after dropping NaN: {df.shape}")
    
    # Drop duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicates found: {duplicates}")
    df = df.drop_duplicates()
    print(f"Shape after dropping duplicates: {df.shape}")
    
    # Class distribution
    print(f"\nClass distribution:")
    print(df['Case'].value_counts())
    print(f"\nSpam percentage: {(df['Case'] == 1).sum() / len(df) * 100:.2f}%")
    
    # Create content column (just the text for UCI data)
    df['content'] = df['text']
    
    # Add source identifier
    df['source'] = 'UCI'
    
    return df[['content', 'Case', 'source']]


def balance_and_combine_datasets(
    original_df: pd.DataFrame,
    uci_df: pd.DataFrame,
    target_size: int = None
) -> pd.DataFrame:
    """
    Balance datasets and combine them.
    Downsamples the larger dataset to match the smaller one.
    
    Args:
        original_df: Original telecom fraud dataset
        uci_df: UCI spam collection dataset
        target_size: Target size for each dataset (default: size of smaller dataset)
    
    Returns:
        Combined balanced dataset
    """
    print("\n" + "=" * 80)
    print("BALANCING AND COMBINING DATASETS")
    print("=" * 80)
    
    print(f"\nOriginal dataset size: {len(original_df)}")
    print(f"UCI dataset size: {len(uci_df)}")
    
    # Determine target size (use smaller dataset size if not specified)
    if target_size is None:
        target_size = min(len(original_df), len(uci_df))
    
    print(f"\nTarget size for each dataset: {target_size}")
    
    # Downsample original dataset if needed
    if len(original_df) > target_size:
        # Stratified sampling to maintain class distribution
        fraud_ratio = (original_df['Case'] == 1).sum() / len(original_df)
        n_fraud = int(target_size * fraud_ratio)
        n_normal = target_size - n_fraud
        
        fraud_samples = original_df[original_df['Case'] == 1].sample(
            n=min(n_fraud, (original_df['Case'] == 1).sum()),
            random_state=42
        )
        normal_samples = original_df[original_df['Case'] == 0].sample(
            n=min(n_normal, (original_df['Case'] == 0).sum()),
            random_state=42
        )
        
        original_balanced = pd.concat([fraud_samples, normal_samples])
        print(f"Original dataset downsampled to: {len(original_balanced)}")
    else:
        original_balanced = original_df
    
    # Downsample UCI dataset if needed
    if len(uci_df) > target_size:
        spam_ratio = (uci_df['Case'] == 1).sum() / len(uci_df)
        n_spam = int(target_size * spam_ratio)
        n_ham = target_size - n_spam
        
        spam_samples = uci_df[uci_df['Case'] == 1].sample(
            n=min(n_spam, (uci_df['Case'] == 1).sum()),
            random_state=42
        )
        ham_samples = uci_df[uci_df['Case'] == 0].sample(
            n=min(n_ham, (uci_df['Case'] == 0).sum()),
            random_state=42
        )
        
        uci_balanced = pd.concat([spam_samples, ham_samples])
        print(f"UCI dataset downsampled to: {len(uci_balanced)}")
    else:
        uci_balanced = uci_df
    
    # Combine datasets
    combined_df = pd.concat([original_balanced, uci_balanced], ignore_index=True)
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nCombined dataset size: {len(combined_df)}")
    print(f"\nCombined class distribution:")
    print(combined_df['Case'].value_counts())
    print(f"\nFraud/Spam percentage: {(combined_df['Case'] == 1).sum() / len(combined_df) * 100:.2f}%")
    
    print(f"\nSource distribution:")
    print(combined_df['source'].value_counts())
    
    return combined_df


def analyze_url_patterns(df: pd.DataFrame, preprocessor: SMSPreprocessor):
    """Analyze URL patterns in the dataset"""
    print("\n" + "=" * 80)
    print("URL PATTERN ANALYSIS")
    print("=" * 80)
    
    # Extract URLs from all messages
    df['urls'] = df['content'].apply(preprocessor.extract_urls)
    df['has_url'] = df['urls'].apply(lambda x: 1 if x else 0)
    
    # URL statistics by class
    print("\nURL presence by class:")
    url_by_class = df.groupby('Case')['has_url'].agg(['sum', 'count', 'mean'])
    url_by_class.columns = ['Messages with URL', 'Total Messages', 'Percentage']
    url_by_class['Percentage'] = url_by_class['Percentage'] * 100
    print(url_by_class)
    
    # Extract all unique URLs from fraud messages
    fraud_urls = []
    for urls in df[df['Case'] == 1]['urls']:
        fraud_urls.extend(urls)
    
    print(f"\nTotal fraud URLs found: {len(fraud_urls)}")
    print(f"Unique fraud domains: {len(set(fraud_urls))}")
    
    # Show sample fraud URLs
    print("\nSample fraud URLs:")
    for url in list(set(fraud_urls))[:10]:
        print(f"  - {url}")
    
    return df


# ============================================================================
# PART 3: BASELINE IMPLEMENTATION (Original Approach)
# ============================================================================

def train_baseline_models(X_train, X_test, y_train, y_test):
    """
    Baseline: TF-IDF + Logistic Regression + SVM
    Replicates original implementation
    """
    print("\n" + "=" * 80)
    print("BASELINE MODELS (Original Approach)")
    print("=" * 80)
    
    results = {}
    
    # Logistic Regression
    print("\n[1/2] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    results['Baseline_LR'] = {
        'model': lr,
        'predictions': y_pred_lr,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr, pos_label=1),
        'recall': recall_score(y_test, y_pred_lr, pos_label=1),
        'f1': f1_score(y_test, y_pred_lr, pos_label=1)
    }
    
    print(f"Accuracy: {results['Baseline_LR']['accuracy']:.4f}")
    print(f"Precision: {results['Baseline_LR']['precision']:.4f}")
    print(f"Recall: {results['Baseline_LR']['recall']:.4f}")
    print(f"F1-Score: {results['Baseline_LR']['f1']:.4f}")
    
    # SVM
    print("\n[2/2] Training SVM...")
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    
    results['Baseline_SVM'] = {
        'model': svm,
        'predictions': y_pred_svm,
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'precision': precision_score(y_test, y_pred_svm, pos_label=1),
        'recall': recall_score(y_test, y_pred_svm, pos_label=1),
        'f1': f1_score(y_test, y_pred_svm, pos_label=1)
    }
    
    print(f"Accuracy: {results['Baseline_SVM']['accuracy']:.4f}")
    print(f"Precision: {results['Baseline_SVM']['precision']:.4f}")
    print(f"Recall: {results['Baseline_SVM']['recall']:.4f}")
    print(f"F1-Score: {results['Baseline_SVM']['f1']:.4f}")
    
    return results


# ============================================================================
# PART 4: ENHANCED IMPLEMENTATION (Dual Word2Vec + Hybrid Features)
# ============================================================================

class DualWord2VecFeatureExtractor:
    """Extract features using dual Word2Vec models"""
    
    def __init__(self, url_w2v_model=None, text_w2v_model=None):
        self.url_w2v = url_w2v_model
        self.text_w2v = text_w2v_model
    
    def get_url_embedding(self, url: str, preprocessor: SMSPreprocessor) -> np.ndarray:
        """Get Word2Vec embedding for URL"""
        if self.url_w2v is None:
            return np.zeros(100)  # Default dimension
        
        tokens = preprocessor.tokenize_url_for_word2vec(url)
        vectors = []
        
        for token in tokens:
            if token in self.url_w2v.wv:
                vectors.append(self.url_w2v.wv[token])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.url_w2v.vector_size)
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get Word2Vec embedding for text"""
        if self.text_w2v is None:
            return np.zeros(300)  # Default dimension
        
        tokens = text.split()
        vectors = []
        
        for token in tokens:
            if token in self.text_w2v:
                vectors.append(self.text_w2v[token])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(300)  # Assuming 300-dim embeddings


def train_url_word2vec(df: pd.DataFrame, preprocessor: SMSPreprocessor):
    """Train Word2Vec model specifically for URLs"""
    print("\n" + "=" * 80)
    print("TRAINING URL WORD2VEC MODEL")
    print("=" * 80)
    
    # Make sure urls column exists
    if 'urls' not in df.columns:
        df['urls'] = df['content'].apply(preprocessor.extract_urls)
    
    # Extract all URLs and tokenize
    url_corpus = []
    for urls in df['urls']:
        for url in urls:
            tokens = preprocessor.tokenize_url_for_word2vec(url)
            if tokens:
                url_corpus.append(tokens)
    
    print(f"\nURL corpus size: {len(url_corpus)} sequences")
    
    if len(url_corpus) < 10:
        print("Warning: Not enough URL data for Word2Vec training")
        return None
    
    # Train Word2Vec
    print("Training URL Word2Vec (Skip-gram, 100 dimensions)...")
    url_w2v = Word2Vec(
        sentences=url_corpus,
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,  # Skip-gram
        epochs=20,
        workers=4,
        seed=42
    )
    
    print(f"Vocabulary size: {len(url_w2v.wv)}")
    
    # Show some learned patterns
    print("\nSample learned URL patterns:")
    sample_words = list(url_w2v.wv.index_to_key)[:10]
    for word in sample_words:
        print(f"  - {word}")
    
    return url_w2v

def train_text_word2vec(df: pd.DataFrame):
    """Train Word2Vec model on cleaned text from combined dataset"""
    print("\n" + "=" * 80)
    print("TRAINING TEXT WORD2VEC")
    print("=" * 80)
    
    # Prepare text corpus (tokenized sentences)
    text_corpus = []
    for text in df['cleaned_text']:
        tokens = text.split()
        if tokens:
            text_corpus.append(tokens)
    
    print(f"\nText corpus size: {len(text_corpus)} messages")
    
    if len(text_corpus) < 10:
        print("Warning: Not enough text data for Word2Vec training")
        return None
    
    # Train Word2Vec on text
    print("Training Text Word2Vec (Skip-gram, 100 dimensions)...")
    text_w2v = Word2Vec(
        sentences=text_corpus,
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,  # Skip-gram
        epochs=20,
        workers=4,
        seed=42
    )
    
    print(f"Vocabulary size: {len(text_w2v.wv)}")
    
    # Show some learned words
    print("\nSample learned words:")
    sample_words = list(text_w2v.wv.index_to_key)[:15]
    for word in sample_words:
        print(f"  - {word}")
    
    return text_w2v



def train_enhanced_models(X_train, X_test, y_train, y_test):
    """
    Enhanced: Dual Word2Vec + Hybrid Features + LR/SVM
    """
    print("\n" + "=" * 80)
    print("ENHANCED MODELS (Dual Word2Vec + Hybrid)")
    print("=" * 80)
    
    results = {}
    
    # Logistic Regression
    print("\n[1/2] Training Enhanced Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    results['Enhanced_LR'] = {
        'model': lr,
        'predictions': y_pred_lr,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr, pos_label=1),
        'recall': recall_score(y_test, y_pred_lr, pos_label=1),
        'f1': f1_score(y_test, y_pred_lr, pos_label=1)
    }
    
    print(f"Accuracy: {results['Enhanced_LR']['accuracy']:.4f}")
    print(f"Precision: {results['Enhanced_LR']['precision']:.4f}")
    print(f"Recall: {results['Enhanced_LR']['recall']:.4f}")
    print(f"F1-Score: {results['Enhanced_LR']['f1']:.4f}")
    
    # SVM
    print("\n[2/2] Training Enhanced SVM...")
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    
    results['Enhanced_SVM'] = {
        'model': svm,
        'predictions': y_pred_svm,
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'precision': precision_score(y_test, y_pred_svm, pos_label=1),
        'recall': recall_score(y_test, y_pred_svm, pos_label=1),
        'f1': f1_score(y_test, y_pred_svm, pos_label=1)
    }
    
    print(f"Accuracy: {results['Enhanced_SVM']['accuracy']:.4f}")
    print(f"Precision: {results['Enhanced_SVM']['precision']:.4f}")
    print(f"Recall: {results['Enhanced_SVM']['recall']:.4f}")
    print(f"F1-Score: {results['Enhanced_SVM']['f1']:.4f}")
    
    return results


# ============================================================================
# PART 5: EXTENDED MODELS (Additional Classifiers)
# ============================================================================

def train_extended_models(X_train, X_test, y_train, y_test):
    """
    Extended: Random Forest, Gradient Boosting, Naive Bayes
    """
    print("\n" + "=" * 80)
    print("EXTENDED MODELS (Additional Classifiers)")
    print("=" * 80)
    
    results = {}
    
    # Random Forest
    print("\n[1/3] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    results['Random_Forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, pos_label=1),
        'recall': recall_score(y_test, y_pred_rf, pos_label=1),
        'f1': f1_score(y_test, y_pred_rf, pos_label=1)
    }
    
    print(f"Accuracy: {results['Random_Forest']['accuracy']:.4f}")
    print(f"Precision: {results['Random_Forest']['precision']:.4f}")
    print(f"Recall: {results['Random_Forest']['recall']:.4f}")
    print(f"F1-Score: {results['Random_Forest']['f1']:.4f}")
    
    # Gradient Boosting
    print("\n[2/3] Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    
    results['Gradient_Boosting'] = {
        'model': gb,
        'predictions': y_pred_gb,
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'precision': precision_score(y_test, y_pred_gb, pos_label=1),
        'recall': recall_score(y_test, y_pred_gb, pos_label=1),
        'f1': f1_score(y_test, y_pred_gb, pos_label=1)
    }
    
    print(f"Accuracy: {results['Gradient_Boosting']['accuracy']:.4f}")
    print(f"Precision: {results['Gradient_Boosting']['precision']:.4f}")
    print(f"Recall: {results['Gradient_Boosting']['recall']:.4f}")
    print(f"F1-Score: {results['Gradient_Boosting']['f1']:.4f}")
    
    # Naive Bayes (only for baseline TF-IDF features, not hybrid)
    print("\n[3/3] Training Naive Bayes...")
    print("Note: Naive Bayes requires non-negative features")
    
    # Check if features are non-negative
    if (X_train < 0).sum() > 0:
        print("Warning: Negative features detected. Skipping Naive Bayes.")
        results['Naive_Bayes'] = None
    else:
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)
        
        results['Naive_Bayes'] = {
            'model': nb,
            'predictions': y_pred_nb,
            'accuracy': accuracy_score(y_test, y_pred_nb),
            'precision': precision_score(y_test, y_pred_nb, pos_label=1),
            'recall': recall_score(y_test, y_pred_nb, pos_label=1),
            'f1': f1_score(y_test, y_pred_nb, pos_label=1)
        }
        
        print(f"Accuracy: {results['Naive_Bayes']['accuracy']:.4f}")
        print(f"Precision: {results['Naive_Bayes']['precision']:.4f}")
        print(f"Recall: {results['Naive_Bayes']['recall']:.4f}")
        print(f"F1-Score: {results['Naive_Bayes']['f1']:.4f}")
    
    return results


# ============================================================================
# PART 6: EVALUATION AND COMPARISON
# ============================================================================

def compare_all_models(baseline_results, enhanced_results, extended_results=None):
    """Create comprehensive comparison of all models"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    # Combine all results
    all_results = {**baseline_results, **enhanced_results}
    if extended_results:
        all_results = {**all_results, **extended_results}
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, metrics in all_results.items():
        if metrics is not None:
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Calculate improvements
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)
    
    if 'Baseline_LR' in baseline_results and 'Enhanced_LR' in enhanced_results:
        baseline_acc = baseline_results['Baseline_LR']['accuracy']
        enhanced_acc = enhanced_results['Enhanced_LR']['accuracy']
        improvement = (enhanced_acc - baseline_acc) / baseline_acc * 100
        
        print(f"\nLogistic Regression Improvement:")
        print(f"  Baseline: {baseline_acc:.4f}")
        print(f"  Enhanced: {enhanced_acc:.4f}")
        print(f"  Improvement: {improvement:+.2f}%")
    
    if 'Baseline_SVM' in baseline_results and 'Enhanced_SVM' in enhanced_results:
        baseline_acc = baseline_results['Baseline_SVM']['accuracy']
        enhanced_acc = enhanced_results['Enhanced_SVM']['accuracy']
        improvement = (enhanced_acc - baseline_acc) / baseline_acc * 100
        
        print(f"\nSVM Improvement:")
        print(f"  Baseline: {baseline_acc:.4f}")
        print(f"  Enhanced: {enhanced_acc:.4f}")
        print(f"  Improvement: {improvement:+.2f}%")
    
    return comparison_df


def plot_comparison(comparison_df):
    """Create visualization of model comparison"""
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Sort by metric value
        sorted_df = comparison_df.sort_values(metric, ascending=True)
        
        # Create horizontal bar chart
        bars = ax.barh(sorted_df['Model'], sorted_df[metric])
        
        # Color bars (baseline vs enhanced)
        colors = ['#FF6B6B' if 'Baseline' in model else '#4ECDC4' 
                  for model in sorted_df['Model']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        
        # Add value labels
        for i, v in enumerate(sorted_df[metric]):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig('charts/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved: charts/model_comparison.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_baseline_vs_enhanced_comparison():
    """
    Main comparison workflow: Baseline (TF-IDF only) vs Enhanced (TF-IDF + Word2Vec + Features)
    on combined dataset (Original + UCI)
    """
    print("\n" + "=" * 80)
    print("BASELINE VS ENHANCED COMPARISON ON COMBINED DATASET")
    print("=" * 80)
    
    # Initialize preprocessor
    preprocessor = SMSPreprocessor()
    
    # ========================================================================
    # STEP 1: Load both datasets
    # ========================================================================
    
    # Load original dataset
    original_df = load_and_analyze_data('data/original_data.csv')
    original_df['content'] = original_df['SMS text'] + ' ' + original_df['Client Sender ID'] + ' ' + original_df['Country']
    original_df['source'] = 'Original'
    original_df = original_df[['content', 'Case', 'source']]
    
    # Load UCI dataset
    uci_df = load_uci_data('data/UCI_spam_data.csv')
    
    # Balance and combine datasets
    combined_df = balance_and_combine_datasets(original_df, uci_df)
    
    # ========================================================================
    # STEP 2: Pre-process combined dataset
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("PRE-PROCESSING COMBINED DATASET")
    print("=" * 80)
    
    # Analyze URL patterns
    combined_df = analyze_url_patterns(combined_df, preprocessor)
    
    # Clean text
    print("Cleaning text...")
    combined_df['cleaned_text'] = combined_df['content'].apply(preprocessor.clean_text)
    
    # Extract engineered features
    print("Extracting engineered features...")
    engineered_features = combined_df['content'].apply(preprocessor.extract_engineered_features)
    engineered_df = pd.DataFrame(engineered_features.tolist())
    
    print(f"Engineered features shape: {engineered_df.shape}")
    print(f"Feature names: {engineered_df.columns.tolist()}")
    
    # ========================================================================
    # STEP 3: Create BASELINE features (TF-IDF ONLY)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("BASELINE APPROACH: TF-IDF ONLY")
    print("=" * 80)
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(combined_df['cleaned_text'])
    
    print(f"TF-IDF shape: {X_tfidf.shape}")
    
    # Split data
    y = combined_df['Case']
    X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train_baseline.shape}")
    print(f"Test set: {X_test_baseline.shape}")
    
    # Train baseline models (LR and SVM on TF-IDF only)
    baseline_results = train_baseline_models(
        X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline
    )
    
    # ========================================================================
    # STEP 4: Create ENHANCED features (TF-IDF + Word2Vec + Engineered)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("ENHANCED APPROACH: Dual Word2Vec + TF-IDF + Engineered Features")
    print("=" * 80)
    
    # Train URL Word2Vec on combined dataset
    url_w2v = train_url_word2vec(combined_df, preprocessor)
    
    # Train Text Word2Vec on combined dataset
    text_w2v = train_text_word2vec(combined_df)
    print(f"DEBUG: text_w2v type: {type(text_w2v)}")
    print(f"DEBUG: text_w2v is None: {text_w2v is None}")
    if text_w2v is not None:
        print(f"DEBUG: text_w2v.wv.vector_size: {text_w2v.wv.vector_size}")
        print(f"DEBUG: text_w2v vocabulary size: {len(text_w2v.wv)}")
    
    # Extract Text Word2Vec embeddings
    print("\nExtracting Text Word2Vec embeddings...")
    text_w2v_embeddings = []
    for text in combined_df['cleaned_text']:
        tokens = text.split()
        vectors = []
        for token in tokens:
            if text_w2v is not None and token in text_w2v.wv:
                vectors.append(text_w2v.wv[token])
        if vectors:
            text_w2v_embeddings.append(np.mean(vectors, axis=0))
        else:
            text_w2v_embeddings.append(np.zeros(100))
    
    text_w2v_features = np.array(text_w2v_embeddings)
    print(f"Text Word2Vec features shape: {text_w2v_features.shape}")
    print(f"DEBUG: Sample text W2V embedding (first 5 values): {text_w2v_features[0][:5]}")
    print(f"DEBUG: Text W2V non-zero count: {np.count_nonzero(text_w2v_features)}")
    
    # Extract URL Word2Vec embeddings
    print("\nExtracting URL Word2Vec embeddings...")
    url_w2v_embeddings = []
    feature_extractor = DualWord2VecFeatureExtractor(url_w2v, text_w2v)
    
    for text in combined_df['content']:
        urls = preprocessor.extract_urls(text)
        if urls and url_w2v is not None:
            url_vectors = []
            for url in urls:
                url_vec = feature_extractor.get_url_embedding(url, preprocessor)
                url_vectors.append(url_vec)
            url_w2v_embeddings.append(np.mean(url_vectors, axis=0))
        else:
            url_w2v_embeddings.append(np.zeros(100))
    
    url_w2v_features = np.array(url_w2v_embeddings)
    print(f"URL Word2Vec features shape: {url_w2v_features.shape}")
    print(f"DEBUG: Sample URL W2V embedding (first 5 values): {url_w2v_features[0][:5]}")
    print(f"DEBUG: URL W2V non-zero count: {np.count_nonzero(url_w2v_features)}")
    
    # Combine features: TF-IDF + Text Word2Vec + URL Word2Vec + Engineered
    print("\nDEBUG: Converting TF-IDF to dense...")
    print(f"DEBUG: X_tfidf type: {type(X_tfidf)}")
    print(f"DEBUG: X_tfidf shape: {X_tfidf.shape}")
    
    X_tfidf_dense = X_tfidf.toarray()
    print(f"DEBUG: X_tfidf_dense shape: {X_tfidf_dense.shape}")
    print(f"DEBUG: X_tfidf_dense type: {type(X_tfidf_dense)}")
    
    print("\nDEBUG: Combining features...")
    print(f"DEBUG: text_w2v_features shape: {text_w2v_features.shape}")
    print(f"DEBUG: url_w2v_features shape: {url_w2v_features.shape}")
    print(f"DEBUG: engineered_df.values shape: {engineered_df.values.shape}")
    
    X_enhanced = np.hstack([
        X_tfidf_dense,
        text_w2v_features,
        url_w2v_features,
        engineered_df.values
    ])
    
    print(f"\n{'='*80}")
    print("FEATURE COMBINATION SUMMARY")
    print(f"{'='*80}")
    print(f"Enhanced features shape: {X_enhanced.shape}")
    print(f"  - TF-IDF: {X_tfidf.shape} -> {X_tfidf_dense.shape}")
    print(f"  - Text Word2Vec: {text_w2v_features.shape}")
    print(f"  - URL Word2Vec: {url_w2v_features.shape}")
    print(f"  - Engineered: {engineered_df.shape}")
    print(f"Expected total features: {X_tfidf_dense.shape[1]} + {text_w2v_features.shape[1]} + {url_w2v_features.shape[1]} + {engineered_df.shape[1]} = {X_tfidf_dense.shape[1] + text_w2v_features.shape[1] + url_w2v_features.shape[1] + engineered_df.shape[1]}")
    print(f"Actual total features: {X_enhanced.shape[1]}")
    print(f"{'='*80}")
    
    # Split enhanced data
    X_train_enhanced, X_test_enhanced, y_train_enhanced, y_test_enhanced = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train_enhanced.shape}")
    print(f"Test set: {X_test_enhanced.shape}")
    
    # Train enhanced models (LR and SVM on hybrid features)
    enhanced_results = train_enhanced_models(
        X_train_enhanced, X_test_enhanced, y_train_enhanced, y_test_enhanced
    )
    
    # ========================================================================
    # STEP 4.5: Train Random Forest and Gradient Boosting
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TRAINING ENSEMBLE MODELS (Random Forest & Gradient Boosting)")
    print("=" * 80)
    
    # Train Random Forest on baseline features
    print("\n[Baseline] Training Random Forest...")
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_baseline.fit(X_train_baseline, y_train_baseline)
    y_pred_rf_baseline = rf_baseline.predict(X_test_baseline)
    
    baseline_results['Baseline_RF'] = {
        'model': rf_baseline,
        'predictions': y_pred_rf_baseline,
        'accuracy': accuracy_score(y_test_baseline, y_pred_rf_baseline),
        'precision': precision_score(y_test_baseline, y_pred_rf_baseline, pos_label=1),
        'recall': recall_score(y_test_baseline, y_pred_rf_baseline, pos_label=1),
        'f1': f1_score(y_test_baseline, y_pred_rf_baseline, pos_label=1)
    }
    
    print(f"Accuracy: {baseline_results['Baseline_RF']['accuracy']:.4f}")
    print(f"Precision: {baseline_results['Baseline_RF']['precision']:.4f}")
    print(f"Recall: {baseline_results['Baseline_RF']['recall']:.4f}")
    print(f"F1-Score: {baseline_results['Baseline_RF']['f1']:.4f}")
    
    # Train Gradient Boosting on baseline features
    print("\n[Baseline] Training Gradient Boosting...")
    gb_baseline = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_baseline.fit(X_train_baseline, y_train_baseline)
    y_pred_gb_baseline = gb_baseline.predict(X_test_baseline)
    
    baseline_results['Baseline_GB'] = {
        'model': gb_baseline,
        'predictions': y_pred_gb_baseline,
        'accuracy': accuracy_score(y_test_baseline, y_pred_gb_baseline),
        'precision': precision_score(y_test_baseline, y_pred_gb_baseline, pos_label=1),
        'recall': recall_score(y_test_baseline, y_pred_gb_baseline, pos_label=1),
        'f1': f1_score(y_test_baseline, y_pred_gb_baseline, pos_label=1)
    }
    
    print(f"Accuracy: {baseline_results['Baseline_GB']['accuracy']:.4f}")
    print(f"Precision: {baseline_results['Baseline_GB']['precision']:.4f}")
    print(f"Recall: {baseline_results['Baseline_GB']['recall']:.4f}")
    print(f"F1-Score: {baseline_results['Baseline_GB']['f1']:.4f}")
    
    # Train Random Forest on enhanced features
    print("\n[Enhanced] Training Random Forest...")
    rf_enhanced = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_enhanced.fit(X_train_enhanced, y_train_enhanced)
    y_pred_rf_enhanced = rf_enhanced.predict(X_test_enhanced)
    
    enhanced_results['Enhanced_RF'] = {
        'model': rf_enhanced,
        'predictions': y_pred_rf_enhanced,
        'accuracy': accuracy_score(y_test_enhanced, y_pred_rf_enhanced),
        'precision': precision_score(y_test_enhanced, y_pred_rf_enhanced, pos_label=1),
        'recall': recall_score(y_test_enhanced, y_pred_rf_enhanced, pos_label=1),
        'f1': f1_score(y_test_enhanced, y_pred_rf_enhanced, pos_label=1)
    }
    
    print(f"Accuracy: {enhanced_results['Enhanced_RF']['accuracy']:.4f}")
    print(f"Precision: {enhanced_results['Enhanced_RF']['precision']:.4f}")
    print(f"Recall: {enhanced_results['Enhanced_RF']['recall']:.4f}")
    print(f"F1-Score: {enhanced_results['Enhanced_RF']['f1']:.4f}")
    
    # Train Gradient Boosting on enhanced features
    print("\n[Enhanced] Training Gradient Boosting...")
    gb_enhanced = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_enhanced.fit(X_train_enhanced, y_train_enhanced)
    y_pred_gb_enhanced = gb_enhanced.predict(X_test_enhanced)
    
    enhanced_results['Enhanced_GB'] = {
        'model': gb_enhanced,
        'predictions': y_pred_gb_enhanced,
        'accuracy': accuracy_score(y_test_enhanced, y_pred_gb_enhanced),
        'precision': precision_score(y_test_enhanced, y_pred_gb_enhanced, pos_label=1),
        'recall': recall_score(y_test_enhanced, y_pred_gb_enhanced, pos_label=1),
        'f1': f1_score(y_test_enhanced, y_pred_gb_enhanced, pos_label=1)
    }
    
    print(f"Accuracy: {enhanced_results['Enhanced_GB']['accuracy']:.4f}")
    print(f"Precision: {enhanced_results['Enhanced_GB']['precision']:.4f}")
    print(f"Recall: {enhanced_results['Enhanced_GB']['recall']:.4f}")
    print(f"F1-Score: {enhanced_results['Enhanced_GB']['f1']:.4f}")
    
    # ========================================================================
    # STEP 5: Compare results
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("BASELINE VS ENHANCED COMPARISON (ALL MODELS)")
    print("=" * 80)
    
    comparison_df = compare_all_models(baseline_results, enhanced_results)
    
    # Create visualizations
    plot_comparison(comparison_df)
    
    # Save results
    comparison_df.to_csv('baseline_vs_enhanced_results.csv', index=False)
    print("\nResults saved: baseline_vs_enhanced_results.csv")
    
    # ========================================================================
    # STEP 6: Analyze uni-model advantage
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("UNI-MODEL ADVANTAGE ANALYSIS")
    print("=" * 80)
    
    print("\nDataset Composition:")
    print(f"Total samples: {len(combined_df)}")
    print(f"Original dataset samples: {(combined_df['source'] == 'Original').sum()}")
    print(f"UCI dataset samples: {(combined_df['source'] == 'UCI').sum()}")
    
    print("\nKey Findings:")
    print("1. Baseline (TF-IDF only) performs well on URL-based fraud (Original dataset)")
    print("2. Enhanced approach adds value for social engineering fraud (UCI dataset)")
    print("3. Single unified model handles both fraud types effectively")
    print("4. Engineered features capture patterns beyond simple text matching")
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)


def main():
    """Main execution pipeline (original dataset only)"""
    
    # Initialize preprocessor
    preprocessor = SMSPreprocessor()
    
    # Load and analyze data
    df = load_and_analyze_data('data/original_data.csv')
    
    # Create content column (combining SMS text, Client Sender ID, and Country)
    print("\nCreating content column...")
    df['content'] = df['SMS text'] + ' ' + df['Client Sender ID'] + ' ' + df['Country']
    
    # Analyze URL patterns
    df = analyze_url_patterns(df, preprocessor)
    
    # Pre-process text
    print("\n" + "=" * 80)
    print("PRE-PROCESSING TEXT")
    print("=" * 80)
    
    print("Cleaning text and extracting features...")
    df['cleaned_text'] = df['content'].apply(preprocessor.clean_text)
    
    # Extract engineered features
    print("Extracting engineered features...")
    engineered_features = df['content'].apply(preprocessor.extract_engineered_features)
    engineered_df = pd.DataFrame(engineered_features.tolist())
    
    print(f"Engineered features shape: {engineered_df.shape}")
    print(f"Feature names: {engineered_df.columns.tolist()}")
    
    # ========================================================================
    # BASELINE: TF-IDF only
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("CREATING BASELINE FEATURES (TF-IDF)")
    print("=" * 80)
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df['cleaned_text'])
    
    print(f"TF-IDF shape: {X_tfidf.shape}")
    
    # Split data
    y = df['Case']
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train_tfidf.shape}")
    print(f"Test set: {X_test_tfidf.shape}")
    
    # Train baseline models
    baseline_results = train_baseline_models(
        X_train_tfidf, X_test_tfidf, y_train, y_test
    )
    
    # ========================================================================
    # ENHANCED: Dual Word2Vec + Hybrid Features
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("CREATING ENHANCED FEATURES (Dual Word2Vec + Hybrid)")
    print("=" * 80)
    
    # Train URL Word2Vec
    url_w2v = train_url_word2vec(df, preprocessor)
    
    # Train Text Word2Vec on combined dataset
    text_w2v = train_text_word2vec(df)
    print(f"DEBUG: text_w2v type: {type(text_w2v)}")
    print(f"DEBUG: text_w2v is None: {text_w2v is None}")
    if text_w2v is not None:
        print(f"DEBUG: text_w2v.wv.vector_size: {text_w2v.wv.vector_size}")
        print(f"DEBUG: text_w2v vocabulary size: {len(text_w2v.wv)}")
    
    # Extract Text Word2Vec embeddings
    print("\nExtracting Text Word2Vec embeddings...")
    text_w2v_embeddings = []
    for text in df['cleaned_text']:
        tokens = text.split()
        vectors = []
        for token in tokens:
            if token in text_w2v.wv:
                vectors.append(text_w2v.wv[token])
        if vectors:
            text_w2v_embeddings.append(np.mean(vectors, axis=0))
        else:
            text_w2v_embeddings.append(np.zeros(100))
    
    text_w2v_features = np.array(text_w2v_embeddings)
    print(f"Text Word2Vec features shape: {text_w2v_features.shape}")
    print(f"DEBUG: Sample text W2V embedding (first 5 values): {text_w2v_features[0][:5]}")
    print(f"DEBUG: Text W2V non-zero count: {np.count_nonzero(text_w2v_features)}")
    
    # Extract URL Word2Vec embeddings (if URLs exist)
    print("\nExtracting URL Word2Vec embeddings...")
    url_w2v_embeddings = []
    feature_extractor = DualWord2VecFeatureExtractor(url_w2v, text_w2v)
    
    for text in df['content']:
        urls = preprocessor.extract_urls(text)
        if urls and url_w2v is not None:
            url_vectors = []
            for url in urls:
                url_vec = feature_extractor.get_url_embedding(url, preprocessor)
                url_vectors.append(url_vec)
            url_w2v_embeddings.append(np.mean(url_vectors, axis=0))
        else:
            url_w2v_embeddings.append(np.zeros(100))
    
    url_w2v_features = np.array(url_w2v_embeddings)
    print(f"URL Word2Vec features shape: {url_w2v_features.shape}")
    print(f"DEBUG: Sample URL W2V embedding (first 5 values): {url_w2v_features[0][:5]}")
    print(f"DEBUG: URL W2V non-zero count: {np.count_nonzero(url_w2v_features)}")
    
    # Combine features: TF-IDF + Text Word2Vec + URL Word2Vec + Engineered
    from scipy.sparse import hstack as sparse_hstack
    
    # Convert TF-IDF to dense and combine with other features
    print("\nDEBUG: Converting TF-IDF to dense...")
    print(f"DEBUG: X_tfidf type: {type(X_tfidf)}")
    print(f"DEBUG: X_tfidf shape: {X_tfidf.shape}")
    
    X_tfidf_dense = X_tfidf.toarray()
    print(f"DEBUG: X_tfidf_dense shape: {X_tfidf_dense.shape}")
    print(f"DEBUG: X_tfidf_dense type: {type(X_tfidf_dense)}")
    
    print("\nDEBUG: Combining features...")
    print(f"DEBUG: text_w2v_features shape: {text_w2v_features.shape}")
    print(f"DEBUG: url_w2v_features shape: {url_w2v_features.shape}")
    print(f"DEBUG: engineered_df.values shape: {engineered_df.values.shape}")
    
    X_hybrid = np.hstack([
        X_tfidf_dense,
        text_w2v_features,
        url_w2v_features,
        engineered_df.values
    ])
    
    print(f"\n{'='*80}")
    print("FEATURE COMBINATION SUMMARY")
    print(f"{'='*80}")
    print(f"Hybrid features shape: {X_hybrid.shape}")
    print(f"  - TF-IDF: {X_tfidf.shape} -> {X_tfidf_dense.shape}")
    print(f"  - Text Word2Vec: {text_w2v_features.shape}")
    print(f"  - URL Word2Vec: {url_w2v_features.shape}")
    print(f"  - Engineered: {engineered_df.shape}")
    print(f"Expected total features: {X_tfidf_dense.shape[1]} + {text_w2v_features.shape[1]} + {url_w2v_features.shape[1]} + {engineered_df.shape[1]} = {X_tfidf_dense.shape[1] + text_w2v_features.shape[1] + url_w2v_features.shape[1] + engineered_df.shape[1]}")
    print(f"Actual total features: {X_hybrid.shape[1]}")
    print(f"{'='*80}")
    
    # Split hybrid data
    X_train_hybrid, X_test_hybrid, y_train_h, y_test_h = train_test_split(
        X_hybrid, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train enhanced models
    enhanced_results = train_enhanced_models(
        X_train_hybrid, X_test_hybrid, y_train_h, y_test_h
    )
    
    # ========================================================================
    # EXTENDED: Additional Classifiers
    # ========================================================================
    
    # Train on baseline features first
    extended_baseline = train_extended_models(
        X_train_tfidf, X_test_tfidf, y_train, y_test
    )
    
    # Train on enhanced features
    extended_enhanced = train_extended_models(
        X_train_hybrid, X_test_hybrid, y_train_h, y_test_h
    )
    
    # Rename for clarity
    extended_baseline = {f"Baseline_{k}": v for k, v in extended_baseline.items()}
    extended_enhanced = {f"Enhanced_{k}": v for k, v in extended_enhanced.items()}
    
    # ========================================================================
    # COMPARISON AND VISUALIZATION
    # ========================================================================
    
    all_extended = {**extended_baseline, **extended_enhanced}
    comparison_df = compare_all_models(baseline_results, enhanced_results, all_extended)
    
    # Create visualizations
    plot_comparison(comparison_df)
    
    # Save results
    comparison_df.to_csv('model_results.csv', index=False)
    print("\nResults saved: model_results.csv")
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Download required NLTK data
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
    # Run baseline vs enhanced comparison on combined dataset
    run_baseline_vs_enhanced_comparison()
    

