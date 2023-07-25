import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

data = pd.read_csv("data/complaints.csv")

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(list(train_data["text"]), truncation=True, padding=True)
test_encodings = tokenizer(list(test_data["text"]), truncation=True, padding=True)

label2id = {label: i for i, label in enumerate(data["category"].unique())}
train_labels = [label2id[label] for label in train_data["category"]]
test_labels = [label2id[label] for label in test_data["category"]]

train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "labels": train_labels})
test_dataset = Dataset.from_dict({"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"], "labels": test_labels})

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label2id))

training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)
trainer.train()
model.save_pretrained("models")
