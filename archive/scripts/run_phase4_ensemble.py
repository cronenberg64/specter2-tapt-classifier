import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
try:
    from config import Config
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import Config

def run_ensemble():
    print("--- STARTING ENSEMBLE (FROM SAVED RESULTS) ---")
    if not os.path.exists("results/phase4_leaderboard.json"):
        print("No leaderboard found.")
        return

    with open("results/phase4_leaderboard.json", 'r') as f:
        results = json.load(f)

    # Sort and Top 2
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    top2 = sorted_results[:2]
    if len(top2) < 2:
        print("Need at least 2 models.")
        return
        
    print(f"Top 2 Models: {[m['model'] for m in top2]}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    models = []
    tokenizers = []
    for m in top2:
        print(f"Loading {m['path']}...")
        models.append(AutoModelForSequenceClassification.from_pretrained(m["path"]).to(device))
        tokenizers.append(AutoTokenizer.from_pretrained(m["path"], use_fast=False))
        
    # Re-load test data consistently
    dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train").train_test_split(test_size=0.2, seed=42)["test"]
    
    # Use config from first model for label mapping checks
    label2id = models[0].config.label2id
    true_labels = [label2id[l] for l in dataset["label"]]
    preds_ensemble = []
    
    print(f"Running Inference on {len(dataset)} samples...")
    with torch.no_grad():
        for i, item in enumerate(dataset):
            text = item["text"]
            probs_sum = None
            for model, tokenizer in zip(models, tokenizers):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=1)
                if probs_sum is None: probs_sum = probs
                else: probs_sum += probs
            
            avg_probs = probs_sum / len(models)
            preds_ensemble.append(torch.argmax(avg_probs).item())
            
    acc = accuracy_score(true_labels, preds_ensemble)
    print(f"Ensemble Accuracy: {acc}")
    
    with open("results/phase4_ensemble_result.json", "w") as f:
        json.dump({"ensemble_accuracy": acc, "top_models": top2}, f, indent=4)

if __name__ == "__main__":
    run_ensemble()
