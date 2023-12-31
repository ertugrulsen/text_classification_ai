import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

data = pd.read_csv("data/sikayetimVar.csv")

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

model_name = "dbmdz/distilbert-base-turkish-cased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(list(train_data["text"]), truncation=True, padding=True)
test_encodings = tokenizer(list(test_data["text"]), truncation=True, padding=True)

label2id = {label: i for i, label in enumerate(train_data["category"].unique())}
train_labels = [label2id[label] for label in train_data["category"]]
test_labels = [label2id[label] for label in test_data["category"]]

train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "label": train_labels})
test_dataset = Dataset.from_dict({"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"], "label": test_labels})

model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id))

training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=500,
    save_steps=1000,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
trainer.train()
model.save_pretrained("models")

