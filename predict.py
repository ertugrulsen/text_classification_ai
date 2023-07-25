import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("models", cache_dir="./cache")  # Model dosyaları ./cache klasörüne indirilecek
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", cache_dir="./cache")  # Tokenizer dosyaları ./cache klasörüne indirilecek
    return model, tokenizer


def classify_text(text, model, tokenizer):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()

    return predicted_class

model, tokenizer = load_model()

sample_texts = [
    "kredi kartımın borcu hatalı olarak gösteriliyor.",
    "atm para çekme işlemi başarısız oldu.",
    "müşteri hizmetleriyle iletişime geçemiyorum."
]

for text in sample_texts:
    predicted_class = classify_text(text, model, tokenizer)
    label2id = {1: "müşteri hizmetleri", 2: "kredi başvurusu", 3: "mobil uygulama", 4: "ücretlendirme",
        5: "atm", 6: "kredi kartı", 7: "internet bankacılığı", 8: "bilgi ve iletişim", 9: "faiz işlem",
                10: "şube hizmeti", 11: "çağrı merkezi", 12: "para transferi", 13: "pos" }
    predicted_category = label2id[predicted_class]
    print(f"Metin: {text}")
    print(f"Tahmin: {predicted_category}\n")