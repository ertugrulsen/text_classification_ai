import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

class TrainClassifier:
    def __init__(self, data_file):
        self.data_file = data_file
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Zğüşıöç\s]', '', text)  # Türkçe karakterler için düzenleme
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('turkish'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)

    def load_data(self):
        data = pd.read_csv(self.data_file, encoding='utf-8')
        data['clean_text'] = data['text'].apply(self.preprocess_text)
        X = data['clean_text']
        y = data['category']
        return X, y

    def tokenize_data(self, text_list):
        return self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

    def train_model(self):
        X, y = self.load_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Tokenize the text data and prepare them as tensors
        train_encodings = self.tokenize_data(list(X_train))
        val_encodings = self.tokenize_data(list(X_val))

        # Prepare the Trainer and TrainingArguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_steps=500,
            save_steps=1000,
            logging_dir="./logs",
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_encodings,
            eval_dataset=val_encodings,
        )

        # Train the model
        trainer.train()

        return trainer

class PredictClassifier:
    def __init__(self, model_file):
        self.model = DistilBertForSequenceClassification.from_pretrained(model_file)
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Zğüşıöç\s]', '', text)  # Türkçe karakterler için düzenleme
        return text

    def predict(self, text):
        text = self.preprocess_text(text)
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax().item()
        return predicted_class

if __name__ == "__main__":
    data_file = 'data/sikayetimVar.csv'  # Yeni CSV dosya yolunu burada ayarlayın
    trainer = TrainClassifier(data_file)
    trained_model = trainer.train_model()

    model_file = "./results/checkpoint-1000"
    predictor = PredictClassifier(model_file)

    complaint1 = "müşteri hizmetleriyle iletişime geçemiyorum"
    complaint2 = "ürünümün teslimatı gecikti"

    predicted_category1 = predictor.predict(complaint1)
    predicted_category2 = predictor.predict(complaint2)

    print("Şikayet 1 Tahmin Edilen Kategori:", predicted_category1)
    print("Şikayet 2 Tahmin Edilen Kategori:", predicted_category2)
