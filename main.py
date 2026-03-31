"""
main.py
Entry point — runs the complete Spam Detection pipeline.

Usage:
  python main.py              → full pipeline (generate → train → evaluate → plot)
  python main.py --demo       → interactive demo mode
  python main.py --email "…"  → classify a single email
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_generator import generate_dataset
from model import SpamDetector, train_and_evaluate
from visualize import generate_all_plots
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse


def run_full_pipeline():
    print("\n🚀  SPAM DETECTION — FULL PIPELINE\n" + "="*50)

    # 1. Generate dataset
    print("\n[1/4] Generating dataset...")
    df = generate_dataset(1200)
    df.to_csv('emails.csv', index=False)
    _, X_test, __, y_test = train_test_split(
        df['email'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    # 2. Train & evaluate all NB variants
    print("\n[2/4] Training Naive Bayes models...")
    best_model, all_results = train_and_evaluate('emails.csv')

    # 3. Generate plots
    print("\n[3/4] Generating evaluation plots...")
    generate_all_plots(best_model, all_results, X_test.tolist(), y_test.tolist())

    # 4. Demo predictions
    print("\n[4/4] Demo predictions:")
    demo_emails = [
        "CONGRATULATIONS! You've been selected to WIN $50,000! Claim your FREE prize NOW!!!",
        "Hey Sarah, are we still on for the 3pm team standup today?",
        "URGENT: Your bank account is compromised. Verify your info at secure-bank.xyz immediately!",
        "Please review the attached Q3 report before tomorrow's board meeting.",
        "Get rich quick! Make $5000/week from home. No experience needed. 100% GUARANTEED!",
        "The project deadline has been moved to Friday. Please update your timelines accordingly.",
    ]

    print("\n" + "-"*70)
    print(f"{'Email (truncated)':<45} {'Label':<8} {'Spam%':<8} {'Confidence'}")
    print("-"*70)
    for email in demo_emails:
        result = best_model.predict_single(email)
        icon = "🔴" if result['is_spam'] else "🟢"
        label = "SPAM" if result['is_spam'] else "HAM"
        print(f"{icon} {email[:42]:<42} {label:<8} {result['spam_probability']:.1%}    {result['confidence']:.1%}")
    print("-"*70)
    print("\n✅  Pipeline complete! Check ./plots/ for visualizations.\n")


def demo_mode():
    """Interactive CLI demo."""
    print("\n🛡️  Spam Detector — Interactive Mode")
    print("   Type an email text to classify. Enter 'quit' to exit.\n")

    if not os.path.exists('spam_model.pkl'):
        print("No saved model found. Training now...")
        df = generate_dataset(1000)
        df.to_csv('emails.csv', index=False)
        train_and_evaluate('emails.csv')

    model = SpamDetector.load('spam_model.pkl')
    while True:
        text = input("\nEmail text: ").strip()
        if text.lower() in ('quit', 'exit', 'q'):
            break
        if not text:
            continue
        result = model.predict_single(text)
        icon = "🔴 SPAM" if result['is_spam'] else "🟢 HAM"
        print(f"  → {icon}")
        print(f"     Spam probability : {result['spam_probability']:.1%}")
        print(f"     Ham probability  : {result['ham_probability']:.1%}")
        print(f"     Confidence       : {result['confidence']:.1%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spam Email Detector — Naive Bayes')
    parser.add_argument('--demo', action='store_true', help='Interactive demo mode')
    parser.add_argument('--email', type=str, help='Classify a single email text')
    args = parser.parse_args()

    if args.email:
        if not os.path.exists('spam_model.pkl'):
            print("Training model first...")
            df = generate_dataset(1000)
            df.to_csv('emails.csv', index=False)
            train_and_evaluate('emails.csv')
        model = SpamDetector.load('spam_model.pkl')
        result = model.predict_single(args.email)
        print(f"\nLabel: {'🔴 SPAM' if result['is_spam'] else '🟢 HAM'}")
        print(f"Spam probability: {result['spam_probability']:.1%}")
    elif args.demo:
        demo_mode()
    else:
        run_full_pipeline()
