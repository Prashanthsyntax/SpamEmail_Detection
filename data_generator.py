"""
data_generator.py
Generates a realistic spam/ham email dataset for training.
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

SPAM_TEMPLATES = [
    "Congratulations! You've won a ${amount} prize. Click here to claim now!",
    "FREE offer! Get {product} absolutely free. Limited time offer. Act now!",
    "URGENT: Your account will be suspended. Verify immediately at {link}",
    "Make ${amount} from home every day! No experience needed. 100% guaranteed.",
    "You have been selected for a special {bank} reward. Claim your ${amount} today.",
    "Dear winner, your email has won the {lottery} lottery. Send details to claim.",
    "Buy {product} at 90% discount! Today only. Order now for free shipping!",
    "ALERT: Unusual login detected. Click {link} to secure your account NOW.",
    "Lose {weight} lbs in {days} days! Miracle pill doctors don't want you to know.",
    "Hot singles in your area are waiting. Click here to meet them tonight!",
    "Your PayPal account is limited. Confirm your identity at {link} immediately.",
    "Investment opportunity: {percent}% returns guaranteed. Join {amount} investors.",
    "FINAL NOTICE: You owe ${amount}. Pay now to avoid legal action.",
    "Work from home and earn ${amount}/week. No skills needed. Start today!",
    "FREE iPhone {model}! You are the {number}th visitor. Claim your prize now.",
    "Increase your income by ${amount} monthly with our proven system.",
    "Your computer has {number} viruses! Download our free scanner immediately.",
    "Exclusive deal for you: {product} at ${amount}. Order before midnight!",
    "Congratulations! You qualify for a ${amount} government grant. Apply now.",
    "Meet {name} who makes ${amount}/day trading crypto. Learn his secret!",
]

HAM_TEMPLATES = [
    "Hi {name}, just wanted to follow up on our meeting scheduled for {day}.",
    "Please find attached the report for {project}. Let me know your thoughts.",
    "The team lunch is on {day} at {time}. Hope to see everyone there!",
    "Quick reminder: the {project} deadline is next {day}. Please submit your work.",
    "Hi {name}, can we reschedule our call to {day} at {time}?",
    "Thanks for your email. I'll review the {document} and get back to you by {day}.",
    "The quarterly results are in. Please check the attached {document} for details.",
    "Hi, I'm reaching out regarding the {position} position at our company.",
    "Your package has been shipped. Expected delivery: {day}. Track at {link}",
    "Reminder: Your subscription renews on {day}. No action needed if you wish to continue.",
    "Meeting notes from {day}: {project} discussed. Next steps assigned to {name}.",
    "Happy birthday, {name}! Hope you have a wonderful day.",
    "Could you please review the {document} before our {day} meeting?",
    "I've shared the {project} presentation with you on Google Drive.",
    "Your appointment is confirmed for {day} at {time} with Dr. {name}.",
    "The monthly newsletter is ready. Read about updates to {product} and more.",
    "Hi {name}, your order #{number} has been processed successfully.",
    "Just a heads up that I'll be out of office from {day} to {day}.",
    "Can you send me the latest version of the {document}? Thanks.",
    "The {project} is on track. We completed {percent}% of the milestones this week.",
]

def fill_template(template):
    replacements = {
        '{amount}': str(random.choice([500, 1000, 5000, 10000, 25000, 50000])),
        '{product}': random.choice(['iPhone', 'laptop', 'gift card', 'vacation package', 'car']),
        '{link}': random.choice(['secure-login.xyz', 'verify-now.net', 'claim-prize.co']),
        '{bank}': random.choice(['Chase', 'Wells Fargo', 'Bank of America', 'Citi']),
        '{lottery}': random.choice(['UK National', 'Euro Millions', 'PowerBall', 'Mega Millions']),
        '{weight}': str(random.randint(10, 50)),
        '{days}': str(random.choice([7, 14, 21, 30])),
        '{percent}': str(random.randint(20, 500)),
        '{model}': random.choice(['15 Pro', '14', '13 Max', '16']),
        '{number}': str(random.randint(100, 9999)),
        '{name}': random.choice(['John', 'Sarah', 'Michael', 'Emma', 'David', 'Lisa']),
        '{day}': random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']),
        '{time}': random.choice(['9:00 AM', '10:30 AM', '2:00 PM', '3:30 PM', '4:00 PM']),
        '{project}': random.choice(['Q4 Review', 'Website Redesign', 'Marketing Campaign', 'Budget Analysis']),
        '{document}': random.choice(['proposal', 'contract', 'report', 'presentation', 'spreadsheet']),
        '{position}': random.choice(['Software Engineer', 'Data Analyst', 'Product Manager', 'Designer']),
    }
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result

def generate_dataset(n_samples=1000):
    emails = []
    labels = []

    n_spam = n_samples // 2
    n_ham = n_samples - n_spam

    for _ in range(n_spam):
        template = random.choice(SPAM_TEMPLATES)
        email = fill_template(template)
        # Add some noise
        if random.random() > 0.7:
            email = email.upper()
        emails.append(email)
        labels.append('spam')

    for _ in range(n_ham):
        template = random.choice(HAM_TEMPLATES)
        email = fill_template(template)
        emails.append(email)
        labels.append('ham')

    df = pd.DataFrame({'email': emails, 'label': labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == '__main__':
    df = generate_dataset(1000)
    df.to_csv('emails.csv', index=False)
    print(f"Dataset generated: {len(df)} emails")
    print(df['label'].value_counts())
