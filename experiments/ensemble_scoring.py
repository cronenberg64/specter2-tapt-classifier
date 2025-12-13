import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from src.config import Config

def run_ensemble():
    print("--- STARTING ENSEMBLE (SOFT VOTING) ---")
    
    # 1. Load the Two Champions
    # Ensure these match what train_shootout.py saves
    path_1 = f"{Config.PHASE4_DIR}/roberta-large"
    path_2 = f"{Config.PHASE4_DIR}/scibert_scivocab_uncased"
    
    if not os.path.exists(path_1) or not os.path.exists(path_2):
        print(f"ERROR: Models not found. Expected:\n  {path_1}\n  {path_2}")
        print("Run train_shootout.py first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models to {device}...")
    
    model1 = AutoModelForSequenceClassification.from_pretrained(path_1).to(device)
    model2 = AutoModelForSequenceClassification.from_pretrained(path_2).to(device)
    
    tokenizer1 = AutoTokenizer.from_pretrained(path_1)
    tokenizer2 = AutoTokenizer.from_pretrained(path_2)
    
    # 2. Load Data
    dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    test_data = dataset["test"]
    
    # Encode labels for verification
    labels_list = dataset["train"].unique("label")
    label2id = {l: i for i, l in enumerate(labels_list)}
    
    # 3. Inference Loop
    preds_ensemble = []
    true_labels = [label2id[l] for l in test_data["label"]]
    
    model1.eval()
    model2.eval()
    
    print(f"Running inference on {len(test_data)} samples...")
    
    with torch.no_grad():
        for i, item in enumerate(test_data):
            text = item["text"]
            
            # Model 1 Prediction
            inputs1 = tokenizer1(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            logits1 = model1(**inputs1).logits
            probs1 = F.softmax(logits1, dim=1)
            
            # Model 2 Prediction
            inputs2 = tokenizer2(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            logits2 = model2(**inputs2).logits
            probs2 = F.softmax(logits2, dim=1)
            
            # FUSION: Average the probabilities
            avg_probs = (probs1 + probs2) / 2.0
            final_pred = torch.argmax(avg_probs).item()
            preds_ensemble.append(final_pred)
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(test_data)}")
            
    # 4. Metrics
    acc = accuracy_score(true_labels, preds_ensemble)
    print(f"\n>>> ENSEMBLE ACCURACY: {acc:.4f}")
    print(classification_report(true_labels, preds_ensemble, target_names=labels_list))
    
    # Save result
    if not os.path.exists("results"):
        os.makedirs("results")
    with open("results/phase4_ensemble_result.txt", "w") as f:
        f.write(f"Ensemble Accuracy: {acc:.4f}\n")
        f.write(classification_report(true_labels, preds_ensemble, target_names=labels_list))

if __name__ == "__main__":
    run_ensemble()
