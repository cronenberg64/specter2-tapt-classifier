import pandas as pd
import torch
import os
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import Config
except ImportError:
    class Config:
        DATA_PATH = "data/raw/scientific_abstracts_dataset.csv"
        SEED = 42

OUTPUT_DIR = "models/phase4_shootout/roberta-large"

def train_roberta():
    print("--- Training RoBERTa-Large ---")
    
    if os.path.exists(os.path.join(OUTPUT_DIR, "config.json")):
        print(f"Model already exists at {OUTPUT_DIR}. Skipping.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Data
    df = pd.read_csv(Config.DATA_PATH)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])
    
    dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=Config.SEED)
    
    # Load Model
    model_name = "roberta-large"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    
    # Training Args
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Smaller batch for large model
        learning_rate=2e-5,
        evaluation_strategy="no",
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"]
    )
    
    print("Training...")
    trainer.train()
    
    # Evaluate
    results = trainer.evaluate()
    print(f"Accuracy: {results.get('eval_accuracy', 'N/A')}")
    
    # Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_roberta()
