import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import load_dataset
from setfit import SetFitModel, SetFitTrainer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from src.config import Config
import torch
import os
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1,
    }

print("--- Phase 3: The Beat BERT Suite ---")

# 1. Load Data
print("Loading dataset...")
dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train")

# We need to map labels to integers for SetFit/Transformers
# Assuming labels are strings in 'label' column
labels = dataset.unique("label")
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}

def encode_labels(example):
    example["label"] = label2id[example["label"]]
    return example

dataset = dataset.map(encode_labels)
# Rename 'label' to 'label' (SetFit expects 'label', Transformers 'labels')
# SetFit works with 'label' column. Transformers Trainer expects 'labels'.
# We will handle this per trainer.

# Split
dataset = dataset.train_test_split(test_size=0.2, seed=Config.SEED)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# --- CHAMPION 1: Specter SetFit (The "Small Data" King) ---
print("\n>>> Training Champion 1: Specter 2 with SetFit...")

# Load Specter specifically as a Sentence Transformer
# Note: allenai/specter2_base is a transformer, SetFit wraps it.
# We use the local converted safetensors version to avoid torch.load vulnerability
model_setfit = SetFitModel.from_pretrained("./models/specter2_base_safetensors")

# SetFitTrainer
trainer_setfit = SetFitTrainer(
    model=model_setfit,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    # loss_class=torch.nn.CosineEmbeddingLoss, # Removing this to use default CosineSimilarityLoss
    # Actually SetFit default loss is CosineSimilarityLoss from sentence_transformers.losses
    # Let's use default by not specifying or string "CosineSimilarityLoss"
    metric="accuracy",
    batch_size=32, # Increased for A4000
    num_iterations=5, # Keep at 5
    num_epochs=1,
    column_mapping={"text": "text", "label": "label"}
)

print("Starting SetFit Training...")
trainer_setfit.train()
metrics_setfit = trainer_setfit.evaluate()
print(f"SetFit Result: {metrics_setfit}")

# SKIP SETFIT FOR NOW (STUCK)
# trainer_setfit.train()
# metrics_setfit = trainer_setfit.evaluate()
# print("Skipping SetFit...")
# metrics_setfit = {"skipped": True}
# print(f"SetFit Result: {metrics_setfit}")


# --- CHAMPION 2: Specter Full Fine-Tune (The "Heavyweight") ---
print("\n>>> Training Champion 2: Specter 2 Full Fine-Tune (Unfrozen)...")

model_full = AutoModelForSequenceClassification.from_pretrained(
    "./models/specter2_base_safetensors", 
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

tokenizer = AutoTokenizer.from_pretrained("./models/specter2_base_safetensors")

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=Config.MAX_LENGTH)

# Prepare datasets for Transformers Trainer (needs 'labels' column)
train_dataset_tf = train_dataset.map(lambda x: {"labels": x["label"]}, remove_columns=["label"])
test_dataset_tf = test_dataset.map(lambda x: {"labels": x["label"]}, remove_columns=["label"])

train_dataset_tf = train_dataset_tf.map(tokenize, batched=True)
test_dataset_tf = test_dataset_tf.map(tokenize, batched=True)

# NO FREEZING - Let it learn!
training_args = TrainingArguments(
    output_dir="./models/specter_full_finetune",
    num_train_epochs=Config.EPOCHS_CLF, # Use same epochs as before (5)
    per_device_train_batch_size=Config.BATCH_SIZE,
    per_device_eval_batch_size=Config.BATCH_SIZE,
    learning_rate=2e-5, # Low learning rate is critical when unfreezing
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

trainer_full = Trainer(
    model=model_full,
    args=training_args,
    train_dataset=train_dataset_tf,
    eval_dataset=test_dataset_tf,
    compute_metrics=compute_metrics
)

trainer_full.train()
metrics_full = trainer_full.evaluate()
print(f"Full Fine-Tune Result: {metrics_full}")

# Save Summary
import json
results = {
    "SetFit": metrics_setfit,
    "Full_FineTune": metrics_full
}
with open("phase_3_results.json", "w") as f:
    json.dump(results, f, indent=4)
