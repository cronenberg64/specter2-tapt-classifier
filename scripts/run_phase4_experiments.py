import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
try:
    from config import Config
except ImportError:
    # Fallback if run from scripts/
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import Config

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

def run_shootout():
    Config.MODEL_CANDIDATES = ["google-bert/bert-base-uncased", "allenai/scibert_scivocab_uncased", "microsoft/deberta-v3-base"]
    print(f"--- STARTING PHASE 4 SHOOTOUT: {len(Config.MODEL_CANDIDATES)} MODELS ---")
    
    # Load Data
    dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    # Labels
    labels_list = dataset["train"].unique("label")
    label2id = {l: i for i, l in enumerate(labels_list)}
    id2label = {i: l for i, l in enumerate(labels_list)}
    
    # Encode Labels
    dataset = dataset.map(lambda x: {"label": label2id[x["label"]]})
    
    results = []

    for model_name in Config.MODEL_CANDIDATES:
        try:
            print(f"\n>>> TRAINING CHAMPION: {model_name}")
            
            # Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=Config.MAX_LENGTH)
            
            tokenized_datasets = dataset.map(tokenize_function, batched=True)
            tokenized_datasets = tokenized_datasets.map(lambda x: {"labels": x["label"]}, remove_columns=["label", "text"]) # Clean columns
            
            # Model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=len(labels_list), id2label=id2label, label2id=label2id
            )
            
            # Paths
            clean_name = model_name.split("/")[-1]
            output_dir = os.path.join(Config.PHASE4_DIR, clean_name)
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=Config.EPOCHS_SHOOTOUT,
                per_device_train_batch_size=Config.BATCH_SIZE,
                learning_rate=Config.LEARNING_RATE,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                report_to="none",
                save_total_limit=1,
                fp16=torch.cuda.is_available()
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["test"],
                compute_metrics=compute_metrics
            )
            
            trainer.train()
            metrics = trainer.evaluate()
            
            results.append({
                "model": model_name,
                "accuracy": metrics["eval_accuracy"],
                "f1": metrics["eval_f1"],
                "path": output_dir
            })
            
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Incremental Save
            if not os.path.exists("results"): os.makedirs("results")
            with open("results/phase4_leaderboard.json", "w") as f:
                json.dump(results, f, indent=4)
            
        except Exception as e:
            print(f"FAILED {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save Leaderboard
    if not os.path.exists("results"): os.makedirs("results")
    with open("results/phase4_leaderboard.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n--- SHOOTOUT COMPLETE ---")
    return results

def run_ensemble(results):
    print("\n--- STARTING ENSEMBLE (SOFT VOTING) ---")
    # Identify Top 2 Models based on Accuracy
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    if len(sorted_results) < 2:
        print("Not enough models for ensemble.")
        return

    top2 = sorted_results[:2]
    print(f"Top 2 Models: {[m['model'] for m in top2]}")
    
    device = Config.DEVICE
    
    # Load Models
    models = []
    tokenizers = []
    for m in top2:
        models.append(AutoModelForSequenceClassification.from_pretrained(m["path"]).to(device))
        tokenizers.append(AutoTokenizer.from_pretrained(m["path"], use_fast=False))
        
    # Load Test Data
    dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)["test"]
    labels_list = dataset.unique("label")
    label2id = {l: i for i, l in enumerate(labels_list)}
    
    true_labels = [label2id[l] for l in dataset["label"]]
    preds_ensemble = []
    
    print("Running Inference...")
    with torch.no_grad():
        for i, item in enumerate(dataset):
            text = item["text"]
            probs_sum = None
            
            for model, tokenizer in zip(models, tokenizers):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=Config.MAX_LENGTH).to(device)
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=1)
                
                if probs_sum is None:
                    probs_sum = probs
                else:
                    probs_sum += probs
            
            avg_probs = probs_sum / len(models)
            preds_ensemble.append(torch.argmax(avg_probs).item())
            
    acc = accuracy_score(true_labels, preds_ensemble)
    print(f"Ensemble Accuracy: {acc}")
    
    with open("results/phase4_ensemble_result.json", "w") as f:
        json.dump({"ensemble_accuracy": acc, "top_models": top2}, f, indent=4)

if __name__ == "__main__":
    results = run_shootout()
    if results:
        run_ensemble(results)
