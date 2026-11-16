
import csv
import re
import math
import pickle
import random

import sys
csv.field_size_limit(sys.maxsize)

from collections import Counter, defaultdict

def tokenize(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

print("Loading dataset...")

texts = []
labels = []

with open("enron_spam_data.csv", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        message = (row["Subject"] + " " + row["Message"]).strip()
        label = row["Spam/Ham"].strip().lower()

        if message and label in ("spam", "ham"):
            texts.append(message)
            labels.append(label)

data = list(zip(texts, labels))
random.seed(42)
random.shuffle(data)

split = int(len(data) * 0.8)
train_data = data[:split]
test_data = data[split:]

word_counts = defaultdict(Counter)
class_counts = Counter()
vocabulary = set()

for text, label in train_data:
    class_counts[label] += 1
    words = tokenize(text)
    word_counts[label].update(words)
    vocabulary.update(words)

total_docs = len(train_data)
vocab_size = len(vocabulary)

def predict(text):
    scores = {}

    for label in ("spam", "ham"):
        score = math.log(class_counts[label] / total_docs)
        denominator = sum(word_counts[label].values()) + vocab_size

        for word in tokenize(text):
            score += math.log(
                (word_counts[label][word] + 1) / denominator
            )

        scores[label] = score

    return max(scores, key=scores.get)

correct = sum(
    predict(text) == actual for text, actual in test_data
)

accuracy = correct / len(test_data) * 100

model_data = {
    "word_counts": dict(word_counts),
    "class_counts": dict(class_counts),
    "vocabulary": list(vocabulary),
    "total_docs": total_docs
}

with open("model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("Total emails:", len(data))
print(f"Accuracy: {accuracy:.2f}%")
print("Model saved successfully!")
