import json
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import torch
import os
import sys

# Path hack for config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import Config
except ImportError:
    class Config:
        DATA_PATH = "data/raw/scientific_abstracts_dataset.csv"
        SEED = 42

def run_contextual_training():
    print("\n--- STARTING PHASE 5: CONTEXTUAL INJECTION TRAINING ---")
    
    # 1. Load the Map
    try:
        with open("data/associativity_map.json", "r") as f:
            assoc_map = json.load(f)
        print(f"Loaded Associativity Map with {len(assoc_map)} entries.")
    except FileNotFoundError:
        print("Error: Associativity Map not found. Run src/build_associativity.py first.")
        return

    # 2. Load Data (Standard Split)
    # Ensure consistent split
    df = pd.read_csv(Config.DATA_PATH)
    # Encode labels
    df["label"] = LabelEncoder().fit_transform(df["label"])
    dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=Config.SEED)
    
    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    # --- THE INNOVATION: Context Injection ---
    def inject_context(examples):
        new_inputs = []
        for text in examples["text"]:
            words = set(text.lower().split()) # Set for fast lookup
            hints = []
            
            # Scan for our "Strong Keywords"
            for word in words:
                if word in assoc_map:
                    entry = assoc_map[word]
                    # Format: "word (Domain) [syn1, syn2]"
                    syn_str = ", ".join(entry['synonyms'])
                    hint = f"{word} -> {entry['domain']} (Syns: {syn_str})"
                    hints.append(hint)
            
            # Use top 3 hints max to avoid confusing the model
            if hints:
                # We limit the hints to ensure we don't truncate the actual abstract
                context_str = " | ".join(hints[:3])
                # Append to input
                new_text = f"{text} [SEP] [HINT] {context_str}"
            else:
                new_text = text
                
            new_inputs.append(new_text)
            
        return tokenizer(new_inputs, truncation=True, padding="max_length", max_length=512)

    print("Injecting domain associations into inputs...")
    tokenized_datasets = dataset.map(inject_context, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    
    # 3. Train DeBERTa (Standard Setup)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    args = TrainingArguments(
        output_dir="./models/deberta_context_aware",
        num_train_epochs=5,              # 5 Epochs to ensure it learns to use the hints
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        fp16=torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    metrics = trainer.evaluate()
    print(f"\n>>> FINAL CONTEXTUAL RESULT: {metrics['eval_accuracy']:.4f}")
    
    # Save Model
    trainer.save_model("./models/deberta_context_aware")
    tokenizer.save_pretrained("./models/deberta_context_aware")
    
    # Save Metrics
    if not os.path.exists("results"): os.makedirs("results")
    with open("results/phase5_context_results.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    run_contextual_training()
