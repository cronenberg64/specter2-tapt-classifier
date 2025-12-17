import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset
from src.config import Config
from sklearn.metrics import accuracy_score, f1_score
import torch

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1,
    }

print("--- ModernBERT Benchmark ---")

# 1. Load Data
print("Loading dataset...")
dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train")
labels = dataset.unique("label")
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}

def encode_labels(example):
    example["label"] = label2id[example["label"]]
    return example

dataset = dataset.map(encode_labels)
dataset = dataset.train_test_split(test_size=0.2, seed=Config.SEED)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# 2. Load ModernBERT
model_id = "answerdotai/ModernBERT-base"
print(f"Loading {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, 
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=Config.MAX_LENGTH)

# Prepare datasets
train_dataset_tf = train_dataset.map(lambda x: {"labels": x["label"]}, remove_columns=["label"])
test_dataset_tf = test_dataset.map(lambda x: {"labels": x["label"]}, remove_columns=["label"])

train_dataset_tf = train_dataset_tf.map(tokenize, batched=True)
test_dataset_tf = test_dataset_tf.map(tokenize, batched=True)

# 3. Train
training_args = TrainingArguments(
    output_dir="./models/modernbert_classifier",
    num_train_epochs=Config.EPOCHS_CLF,
    per_device_train_batch_size=Config.BATCH_SIZE, # 8 or 32? Let's stick to 8 to be safe/comparable, or 16. ModernBERT is efficient.
    per_device_eval_batch_size=Config.BATCH_SIZE,
    learning_rate=5e-5, # Standard for BERTs
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=10,
    seed=Config.SEED,
    fp16=(Config.DEVICE == "cuda"),
    use_cpu=(Config.DEVICE == "cpu"),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tf,
    eval_dataset=test_dataset_tf,
    compute_metrics=compute_metrics
)

trainer.train()
metrics = trainer.evaluate()
print(f"ModernBERT Result: {metrics}")

# Save Result
import json
with open("modernbert_results.json", "w") as f:
    json.dump(metrics, f, indent=4)
