import json
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from src.config import Config

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def run_contextual_training():
    print("\n--- STARTING PHASE 5: CONTEXTUAL INJECTION TRAINING ---")
    
    # 1. Load the Map
    try:
        with open(Config.ASSOCIATIVITY_MAP_PATH, "r") as f:
            assoc_map = json.load(f)
    except FileNotFoundError:
        print("Error: Associativity Map not found. Run src/build_associativity.py first.")
        return

    # 2. Load Data (Standard Split)
    # Proactive Fix: Ensure labels are encoded
    df = pd.read_csv(Config.DATA_PATH)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.class_encode_column("label")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    
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
    
    # 3. Train DeBERTa (Standard Setup)
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=3)
    
    args = TrainingArguments(
        output_dir="./models/deberta_context_aware",
        num_train_epochs=5,              # 5 Epochs to ensure it learns to use the hints
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        eval_strategy="epoch",           # Proactive Fix: eval_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics  # Proactive Fix: compute_metrics
    )
    
    trainer.train()
    trainer.save_model() # Save the best model (loaded at end)
    metrics = trainer.evaluate()
    print(f"\n>>> FINAL CONTEXTUAL RESULT: {metrics['eval_accuracy']:.4f}")

if __name__ == "__main__":
    run_contextual_training()
