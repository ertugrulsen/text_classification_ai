from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)

def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("models", cache_dir="./cache")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", cache_dir="./cache")
    return model, tokenizer

model, tokenizer = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.data.decode('utf-8')
    encoding = tokenizer(data, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()

    label2id = {0: "kredi", 1: "kredi kartı", 2: "atm", 3: "müşteri hizmetleri"}
    predicted_category = label2id[predicted_class]

    return predicted_category

@app.route('/')
def homepage():
    return "Hello, This is ai service"

if __name__ == '__main__':
    app.run(debug=True)