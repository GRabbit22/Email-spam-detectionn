
from flask import Flask, render_template, request
import pickle
import re
import math

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model_data = pickle.load(file)

word_counts = model_data["word_counts"]
class_counts = model_data["class_counts"]
vocab_size = len(model_data["vocabulary"])
total_docs = model_data["total_docs"]

def tokenize(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

def predict_email(text):
    scores = {}

    for label in ("spam", "ham"):
        score = math.log(class_counts[label] / total_docs)
        denominator = sum(word_counts[label].values()) + vocab_size

        for word in tokenize(text):
            count = word_counts[label].get(word, 0)
            score += math.log((count + 1) / denominator)

        scores[label] = score

    return max(scores, key=scores.get)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    email_text = ""

    if request.method == "POST":
        email_text = request.form.get("email", "").strip()

        if email_text:
            prediction = predict_email(email_text)

    return render_template(
        "index.html",
        prediction=prediction,
        email_text=email_text
    )

if __name__ == "__main__":
    app.run(debug=True)
