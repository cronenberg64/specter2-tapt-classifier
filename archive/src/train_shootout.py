import json
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
from src.config import Config

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def run_shootout():
    # Load Data (Standard 80/20 split)
    dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train")
    dataset = dataset.class_encode_column("label")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    results = []

    print(f"--- STARTING PHASE 4 SHOOTOUT: {len(Config.MODEL_CANDIDATES)} MODELS ---")

    for model_name in Config.MODEL_CANDIDATES:
        print(f"\n>>> TRAINING CHAMPION: {model_name}")
        
        # 1. Initialize Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Tokenization Helper
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
            
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        # 2. Setup Trainer (Full Fine-Tuning)
        clean_name = model_name.split("/")[-1]
        output_dir = f"{Config.PHASE4_DIR}/{clean_name}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=Config.EPOCHS_SHOOTOUT,
            per_device_train_batch_size=Config.BATCH_SIZE,
            learning_rate=Config.LEARNING_RATE,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            compute_metrics=compute_metrics,
        )

        # 3. Train
        trainer.train()
        
        # 4. Evaluate & Save Stats
        metrics = trainer.evaluate()
        results.append({
            "model": model_name,
            "accuracy": metrics["eval_accuracy"],
            "loss": metrics["eval_loss"],
            "path": output_dir
        })
        
        # SAVE MODEL FOR LATER (Viz & Ensemble)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    # Dump Leaderboard
    with open("phase4_leaderboard.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n--- SHOOTOUT COMPLETE. CHECK phase4_leaderboard.json ---")

if __name__ == "__main__":
    run_shootout()
