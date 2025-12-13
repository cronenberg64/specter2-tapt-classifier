import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from src.config import Config

def run_shootout():
    # Load Data (Standard 80/20 split)
    dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    # Encode labels
    labels = dataset["train"].unique("label")
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    
    def encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example

    dataset = dataset.map(encode_labels)
    
    results = []

    print(f"--- STARTING PHASE 4 SHOOTOUT: {len(Config.MODEL_CANDIDATES)} MODELS ---")

    for model_name in Config.MODEL_CANDIDATES:
        try:
            print(f"\n>>> TRAINING CHAMPION: {model_name}")
            
            # 1. Initialize Tokenizer & Model
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("Tokenizer loaded.")
            
            # Tokenization Helper
            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=Config.MAX_LENGTH)
                
            print("Tokenizing dataset...")
            tokenized_datasets = dataset.map(tokenize_function, batched=True)
            print("Dataset tokenized.")
            
            # Rename label column for Trainer
            tokenized_datasets = tokenized_datasets.map(lambda x: {"labels": x["label"]}, remove_columns=["label"])
            
            print("Loading model...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=len(labels),
                id2label=id2label,
                label2id=label2id
            )
            print("Model loaded.")

            # 2. Setup Trainer (Full Fine-Tuning)
            clean_name = model_name.split("/")[-1]
            output_dir = f"{Config.PHASE4_DIR}/{clean_name}"
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=Config.EPOCHS_SHOOTOUT,
                per_device_train_batch_size=Config.BATCH_SIZE,
                learning_rate=Config.LEARNING_RATE,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy", # Or F1, but accuracy is simpler for shootout
                report_to="none",
                fp16=(Config.DEVICE == "cuda"),
                use_cpu=(Config.DEVICE == "cpu"),
                logging_steps=10
            )
            
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

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["test"],
                compute_metrics=compute_metrics
            )

            # 3. Train
            print("Starting training...")
            trainer.train()
            print("Training complete.")
            
            # 4. Evaluate & Save Stats
            metrics = trainer.evaluate()
            results.append({
                "model": model_name,
                "accuracy": metrics["eval_accuracy"],
                "f1": metrics["eval_f1"],
                "loss": metrics["eval_loss"],
                "path": output_dir
            })
            
            # SAVE MODEL FOR LATER (Viz & Ensemble)
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")
            
        except Exception as e:
            print(f"ERROR TRAINING {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Dump Leaderboard
    if not os.path.exists("results"):
        os.makedirs("results")
        
    with open("results/phase4_leaderboard.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n--- SHOOTOUT COMPLETE. CHECK results/phase4_leaderboard.json ---")

if __name__ == "__main__":
    run_shootout()
