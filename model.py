"""
model.py
Naive Bayes model training, cross-validation, and evaluation.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from preprocessing import EmailPreprocessor


class SpamDetector:
    """
    End-to-end spam detection pipeline using Naive Bayes.

    Supports three Naive Bayes variants:
      - MultinomialNB  → good for raw word counts
      - ComplementNB   → better for imbalanced text data
      - GaussianNB     → handles continuous features
    """

    MODEL_VARIANTS = {
        'multinomial': MultinomialNB,
        'complement': ComplementNB,
        'gaussian': GaussianNB,
    }

    def __init__(self, variant='complement', alpha=1.0):
        self.variant = variant
        self.alpha = alpha
        self.preprocessor = EmailPreprocessor(use_tfidf=True, max_features=3000)
        self._build_model()
        self.is_fitted = False
        self.metrics = {}

    def _build_model(self):
        cls = self.MODEL_VARIANTS.get(self.variant, ComplementNB)
        if self.variant == 'gaussian':
            self.model = cls()
        else:
            self.model = cls(alpha=self.alpha)

    def fit(self, X_text, y):
        """Train the model."""
        X = self.preprocessor.fit_transform(X_text)
        # GaussianNB can handle negative values; MNB/CNB require non-negative
        if self.variant != 'gaussian':
            X = np.abs(X)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, texts):
        """Predict labels for a list of texts."""
        X = self.preprocessor.transform(texts)
        if self.variant != 'gaussian':
            X = np.abs(X)
        return self.model.predict(X)

    def predict_proba(self, texts):
        """Return probability estimates [P(ham), P(spam)]."""
        X = self.preprocessor.transform(texts)
        if self.variant != 'gaussian':
            X = np.abs(X)
        return self.model.predict_proba(X)

    def predict_single(self, email_text: str):
        """Classify a single email and return detailed result."""
        proba = self.predict_proba([email_text])[0]
        label = self.model.classes_[np.argmax(proba)]
        spam_idx = list(self.model.classes_).index('spam')
        return {
            'label': label,
            'is_spam': label == 'spam',
            'spam_probability': float(proba[spam_idx]),
            'ham_probability': float(proba[1 - spam_idx]),
            'confidence': float(max(proba)),
        }

    def evaluate(self, X_text, y_true, cv_folds=5):
        """Full evaluation: hold-out + cross-validation."""
        X = self.preprocessor.transform(X_text)
        if self.variant != 'gaussian':
            X = np.abs(X)

        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, list(self.model.classes_).index('spam')]

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label='spam'),
            'recall': recall_score(y_true, y_pred, pos_label='spam'),
            'f1': f1_score(y_true, y_pred, pos_label='spam'),
            'roc_auc': roc_auc_score((np.array(y_true) == 'spam').astype(int), y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'report': classification_report(y_true, y_pred),
        }
        self.metrics = metrics
        return metrics

    def cross_validate(self, X_text, y, cv=5):
        """Run stratified K-fold cross-validation."""
        X = self.preprocessor.fit_transform(X_text)
        if self.variant != 'gaussian':
            X = np.abs(X)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = {}
        for metric in ['accuracy', 'f1_weighted', 'roc_auc']:
            sc = cross_val_score(self.model, X, y, cv=skf, scoring=metric)
            scores[metric] = {'mean': sc.mean(), 'std': sc.std(), 'all': sc.tolist()}
        self.is_fitted = True
        return scores

    def save(self, path='spam_model.pkl'):
        """Persist the trained model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'preprocessor': self.preprocessor,
                         'variant': self.variant, 'metrics': self.metrics}, f)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path='spam_model.pkl'):
        """Load a saved model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        detector = cls(variant=data['variant'])
        detector.model = data['model']
        detector.preprocessor = data['preprocessor']
        detector.metrics = data['metrics']
        detector.is_fitted = True
        return detector


def train_and_evaluate(csv_path='emails.csv'):
    """Full training pipeline with comparison across NB variants."""
    print("=" * 60)
    print("  SPAM EMAIL DETECTION — NAIVE BAYES")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"\nDataset: {len(df)} emails | Spam: {(df.label=='spam').sum()} | Ham: {(df.label=='ham').sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df['email'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    results = {}
    for variant in ['multinomial', 'complement', 'gaussian']:
        print(f"Training {variant.upper()} Naive Bayes...")
        detector = SpamDetector(variant=variant)
        detector.fit(X_train.tolist(), y_train.tolist())
        metrics = detector.evaluate(X_test.tolist(), y_test.tolist())
        results[variant] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['roc_auc']:.4f}")

    # Pick the best model (by F1)
    best_variant = max(results, key=lambda v: results[v]['f1'])
    print(f"\nBest variant: {best_variant.upper()} (F1={results[best_variant]['f1']:.4f})")

    # Retrain best model and save
    best = SpamDetector(variant=best_variant)
    best.fit(X_train.tolist(), y_train.tolist())
    best.save('spam_model.pkl')

    print("\nClassification Report:")
    print(results[best_variant]['report'])
    return best, results


if __name__ == '__main__':
    from data_generator import generate_dataset
    df = generate_dataset(1000)
    df.to_csv('emails.csv', index=False)
    best_model, all_results = train_and_evaluate('emails.csv')

    # Demo predictions
    test_emails = [
        "CONGRATULATIONS! You've WON $10,000 FREE! Claim NOW!!!",
        "Hi John, can we reschedule Thursday's meeting to Friday at 2pm?",
        "Your account has been compromised. Click here immediately to secure it!",
        "Please find attached the quarterly report for your review.",
    ]
    print("\n--- Live Predictions ---")
    for email in test_emails:
        result = best_model.predict_single(email)
        status = "🔴 SPAM" if result['is_spam'] else "🟢 HAM"
        print(f"{status} ({result['spam_probability']:.1%}) | {email[:60]}...")
