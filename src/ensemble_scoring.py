import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from src.config import Config

def run_ensemble():
    print("--- STARTING ENSEMBLE (SOFT VOTING) ---")
    
    # 1. Load the Two Champions
    path_1 = f"{Config.PHASE4_DIR}/deberta-v3-base"
    path_2 = f"{Config.PHASE4_DIR}/scibert_scivocab_uncased"
    
    try:
        model1 = AutoModelForSequenceClassification.from_pretrained(path_1).to(Config.DEVICE)
        model2 = AutoModelForSequenceClassification.from_pretrained(path_2).to(Config.DEVICE)
        
        tokenizer1 = AutoTokenizer.from_pretrained(path_1)
        tokenizer2 = AutoTokenizer.from_pretrained(path_2)
    except Exception as e:
        print(f"Could not load models for ensemble: {e}")
        return
    
    # 2. Load Data
    # 2. Load Data
    test_data = load_dataset("csv", data_files=Config.DATA_PATH, split="train").class_encode_column("label").train_test_split(test_size=0.2, seed=42)["test"]
    
    # 3. Inference Loop
    preds_ensemble = []
    true_labels = test_data["label"]
    
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        for item in test_data:
            text = item["text"]
            
            # Model 1 Prediction
            inputs1 = tokenizer1(text, return_tensors="pt", truncation=True, max_length=512).to(Config.DEVICE)
            logits1 = model1(**inputs1).logits
            probs1 = F.softmax(logits1, dim=1)
            
            # Model 2 Prediction
            inputs2 = tokenizer2(text, return_tensors="pt", truncation=True, max_length=512).to(Config.DEVICE)
            logits2 = model2(**inputs2).logits
            probs2 = F.softmax(logits2, dim=1)
            
            # FUSION: Average the probabilities
            avg_probs = (probs1 + probs2) / 2.0
            final_pred = torch.argmax(avg_probs).item()
            preds_ensemble.append(final_pred)
            
    # 4. Metrics
    acc = accuracy_score(true_labels, preds_ensemble)
    print(f"\n>>> ENSEMBLE ACCURACY: {acc:.4f}")
    print(classification_report(true_labels, preds_ensemble))

if __name__ == "__main__":
    run_ensemble()
