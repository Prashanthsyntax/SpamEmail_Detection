"""
preprocessing.py
Text preprocessing pipeline for spam email detection.
"""

import re
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class EmailPreprocessor:
    """
    Handles all text cleaning and feature extraction for emails.
    """

    # Common English stop words (lightweight, no NLTK dependency)
    STOP_WORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
        'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
        'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
        'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these',
        'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
        'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
        'against', 'between', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
        'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
        "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
        'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
        "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
        'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
        'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
        "wouldn't"
    }

    def __init__(self, use_tfidf=True, max_features=3000, ngram_range=(1, 2)):
        self.use_tfidf = use_tfidf
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None

    def clean_text(self, text: str) -> str:
        """Apply all text cleaning steps."""
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', ' url ', text)

        # Replace email addresses
        text = re.sub(r'\S+@\S+', ' email ', text)

        # Replace numbers/currency patterns
        text = re.sub(r'\$[\d,]+', ' moneysign ', text)
        text = re.sub(r'\b\d+%', ' percent ', text)
        text = re.sub(r'\b\d+\b', ' number ', text)

        # Remove punctuation (keep spaces)
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove stop words and short tokens
        tokens = [w for w in text.split() if w not in self.STOP_WORDS and len(w) > 1]

        return ' '.join(tokens)

    def extract_features(self, texts):
        """Extract hand-crafted features."""
        features = []
        for text in texts:
            raw = text if isinstance(text, str) else ''
            features.append({
                'num_exclamation': raw.count('!'),
                'num_uppercase': sum(1 for c in raw if c.isupper()),
                'uppercase_ratio': sum(1 for c in raw if c.isupper()) / max(len(raw), 1),
                'has_url': int(bool(re.search(r'http|www|\.com|\.net|\.xyz', raw, re.I))),
                'has_money': int(bool(re.search(r'\$|\bfree\b|\bwin\b|\bprize\b', raw, re.I))),
                'text_length': len(raw),
                'num_digits': sum(c.isdigit() for c in raw),
                'num_special': sum(c in '!$%#@*' for c in raw),
            })
        return np.array([[v for v in f.values()] for f in features], dtype=float)

    def fit_transform(self, texts):
        """Fit vectorizer and transform texts."""
        cleaned = [self.clean_text(t) for t in texts]
        VectClass = TfidfVectorizer if self.use_tfidf else CountVectorizer
        self.vectorizer = VectClass(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=2
        )
        tfidf_features = self.vectorizer.fit_transform(cleaned).toarray()
        hand_features = self.extract_features(texts)
        return np.hstack([tfidf_features, hand_features])

    def transform(self, texts):
        """Transform texts using already-fitted vectorizer."""
        if self.vectorizer is None:
            raise RuntimeError("Call fit_transform first.")
        cleaned = [self.clean_text(t) for t in texts]
        tfidf_features = self.vectorizer.transform(cleaned).toarray()
        hand_features = self.extract_features(texts)
        return np.hstack([tfidf_features, hand_features])


if __name__ == '__main__':
    sample = ["Congratulations! You've WON $5000 FREE prize!", "Hi Sarah, meeting at 3pm today."]
    pp = EmailPreprocessor()
    X = pp.fit_transform(sample)
    print(f"Feature matrix shape: {X.shape}")
    for i, t in enumerate(sample):
        print(f"\n[{i}] Original: {t}")
        print(f"    Cleaned:  {pp.clean_text(t)}")
