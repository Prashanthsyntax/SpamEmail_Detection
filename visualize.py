"""
visualize.py
Generates evaluation plots: confusion matrix, ROC curve, feature importance.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ── Styling ──────────────────────────────────────────────────────────────────
PALETTE = {'spam': '#e74c3c', 'ham': '#2ecc71', 'bg': '#1a1a2e', 'card': '#16213e',
           'accent': '#0f3460', 'text': '#e0e0e0', 'grid': '#2d2d4e'}
plt.rcParams.update({'figure.facecolor': PALETTE['bg'], 'axes.facecolor': PALETTE['card'],
                     'text.color': PALETTE['text'], 'axes.labelcolor': PALETTE['text'],
                     'xtick.color': PALETTE['text'], 'ytick.color': PALETTE['text'],
                     'axes.edgecolor': PALETTE['grid'], 'grid.color': PALETTE['grid'],
                     'font.family': 'monospace'})


def plot_confusion_matrix(cm, classes=['ham', 'spam'], save_path='plots/confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=classes, yticklabels=classes,
                linewidths=2, linecolor=PALETTE['bg'],
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curve(y_true, y_scores, save_path='plots/roc_curve.png'):
    y_bin = (np.array(y_true) == 'spam').astype(int)
    fpr, tpr, _ = roc_curve(y_bin, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=PALETTE['spam'], lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color=PALETTE['grid'], lw=2, linestyle='--', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE['spam'])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — Spam Detection', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_comparison(all_results, save_path='plots/metrics_comparison.png'):
    variants = list(all_results.keys())
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    x = np.arange(len(metrics_keys))
    width = 0.25
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (variant, color) in enumerate(zip(variants, colors)):
        values = [all_results[variant][m] for m in metrics_keys]
        bars = ax.bar(x + i * width, values, width, label=variant.upper(),
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, color=PALETTE['text'])

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Naive Bayes Variants — Performance Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace('_', ' ').upper() for m in metrics_keys])
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_spam_probability_distribution(proba_spam, y_true, save_path='plots/probability_dist.png'):
    spam_mask = np.array(y_true) == 'spam'
    ham_mask = ~spam_mask

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(proba_spam[ham_mask], bins=bins, color=PALETTE['ham'], alpha=0.7, label='Ham Emails', density=True)
    ax.hist(proba_spam[spam_mask], bins=bins, color=PALETTE['spam'], alpha=0.7, label='Spam Emails', density=True)
    ax.axvline(x=0.5, color='white', linestyle='--', lw=2, label='Decision Boundary (0.5)')
    ax.set_xlabel('Predicted Spam Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Spam Probability Distribution', fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_top_features(model, vectorizer, n=20, save_path='plots/top_features.png'):
    """Plot top spam vs ham indicator words."""
    try:
        feature_names = vectorizer.get_feature_names_out()
        classes = list(model.classes_)
        spam_idx = classes.index('spam')
        ham_idx = classes.index('ham')
        log_probs = model.feature_log_prob_

        spam_scores = log_probs[spam_idx][:len(feature_names)]
        ham_scores = log_probs[ham_idx][:len(feature_names)]
        diff = spam_scores - ham_scores

        top_spam_idx = diff.argsort()[-n:][::-1]
        top_ham_idx = diff.argsort()[:n]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        for ax, indices, title, color in [
            (ax1, top_spam_idx, 'Top Spam Indicators', PALETTE['spam']),
            (ax2, top_ham_idx, 'Top Ham Indicators', PALETTE['ham']),
        ]:
            words = [feature_names[i] for i in indices]
            scores = [abs(diff[i]) for i in indices]
            y_pos = range(len(words))
            ax.barh(y_pos, scores, color=color, alpha=0.85, edgecolor='white', linewidth=0.4)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words, fontsize=10)
            ax.set_xlabel('Log-Probability Difference', fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()

        plt.suptitle('Most Informative Features', fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Could not plot top features: {e}")


def generate_all_plots(best_model, all_results, X_test, y_test):
    import os
    os.makedirs('plots', exist_ok=True)

    cm = best_model.metrics.get('confusion_matrix', [[0, 0], [0, 0]])
    plot_confusion_matrix(cm)

    X_feat = best_model.preprocessor.transform(X_test)
    if best_model.variant != 'gaussian':
        X_feat = abs(X_feat)
    spam_idx = list(best_model.model.classes_).index('spam')
    y_scores = best_model.model.predict_proba(X_feat)[:, spam_idx]

    plot_roc_curve(y_test, y_scores)
    plot_metrics_comparison(all_results)
    plot_spam_probability_distribution(y_scores, y_test)

    if hasattr(best_model.model, 'feature_log_prob_'):
        plot_top_features(best_model.model, best_model.preprocessor.vectorizer)

    print("\nAll plots saved to ./plots/")


if __name__ == '__main__':
    from data_generator import generate_dataset
    from model import train_and_evaluate
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = generate_dataset(1000)
    df.to_csv('emails.csv', index=False)
    _, X_test, __, y_test = train_test_split(df['email'], df['label'], test_size=0.2,
                                             random_state=42, stratify=df['label'])
    best_model, all_results = train_and_evaluate('emails.csv')
    generate_all_plots(best_model, all_results, X_test.tolist(), y_test.tolist())
