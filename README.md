
# MailGuard AI – Email Spam Detection System

An AI-powered email spam detection web application built using Python and Flask.

## Features
- Detects spam and legitimate emails.
- Machine Learning classification using Multinomial Naive Bayes.
- Trained on the Enron email dataset.
- Simple and responsive web interface.
- Instant email classification.

## Technology Stack
- Python
- Flask
- Machine Learning
- Multinomial Naive Bayes
- HTML
- CSS
- CSV and Pickle

## Model Performance
- Dataset: 33,665 emails
- Train-test split: 80:20
- Reported test accuracy: 98.80%

## Project Structure
```text
Email-spam-detection/
├── app.py
├── train_model.py
├── model.pkl
├── requirements.txt
├── README.md
├── .gitignore
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## How to Run

1. Clone this repository.
2. Install dependencies:

   pip install -r requirements.txt

3. Start the application:

   python app.py

4. Open http://127.0.0.1:5000 in your browser.

## Author
Garima Gupta
