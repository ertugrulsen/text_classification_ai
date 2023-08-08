import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

class TrainClassifier:
    def __init__(self, model_name, train_file, output_dir="./results", num_train_epochs=10, per_device_train_batch_size=2, per_device_eval_batch_size=2, logging_steps=500, save_steps=1000, logging_dir="./logs"):
        self.model_name = model_name
        self.train_file = train_file
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.logging_dir = logging_dir
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.test_dataset = None
        self.label2id = None

    def preprocess_text(self, text):
        # Metni küçük harfe dönüştürme
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        stop_words_list = ["ve", "veya", ...]
        text = " ".join([word for word in text.split() if word not in stop_words_list])


        return text

    def load_data(self):
        data = pd.read_csv(self.train_file)
        data["text"] = data["text"].apply(self.preprocess_text)  # Metin verilerini ön işleme yap
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        train_encodings = self.tokenizer(list(train_data["text"]), truncation=True, padding=True)
        test_encodings = self.tokenizer(list(test_data["text"]), truncation=True, padding=True)

        self.label2id = {label: i for i, label in enumerate(train_data["category"].unique())}
        train_labels = [self.label2id[label] for label in train_data["category"]]
        test_labels = [self.label2id[label] for label in test_data["category"]]

        self.train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "label": train_labels})
        self.test_dataset = Dataset.from_dict({"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"], "label": test_labels})

    def train_model(self):
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name, num_labels=len(self.label2id))

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            logging_dir=self.logging_dir,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
        )
        trainer.train()
        self.model.save_pretrained(self.output_dir)

    def main(self):
        self.load_data()
        self.train_model()

if __name__ == "__main__":
    model_name = "dbmdz/distilbert-base-turkish-cased"
    train_file = "data/sikayetimVar.csv"
    output_dir = "./results"
    num_train_epochs = 40
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    logging_steps = 500
    save_steps = 1000
    logging_dir = "./logs"

    classifier = TrainClassifier(model_name, train_file, output_dir, num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size, logging_steps, save_steps, logging_dir)
    classifier.main()
