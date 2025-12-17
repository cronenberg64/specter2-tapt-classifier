import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import pandas as pd
import json
import sys
import os
# Add project root to sys.path to allow running script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import Config

def run_synergy_ensemble():
    print("--- STARTING PHASE 6: CONFIDENCE-BASED SYNERGY ENSEMBLE ---")
    
    # 1. Load Data (Strict Test Set)
    # Proactive Fix: Ensure label encoding
    dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train")
    dataset = dataset.class_encode_column("label")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)["test"]
    
    # 2. Load Models
    # Path to Phase 4 Winner (Domain Expert)
    path_sci = "./models/phase4_shootout/scibert_scivocab_uncased" 
    # Path to Phase 5 Winner (Context Expert)
    path_deb = "./models/deberta_context_aware"     
    
    print("Loading Models...")
    print("Loading Models...")
    # Force try CUDA
    try:
        dummy = torch.tensor([1.0]).to("cuda")
        device = "cuda"
        print(f"SUCCESS: CUDA is working. Device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        device = "cpu"
        print(f"ERROR: Failed to use CUDA. Error details: {e}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    print(f"Using device: {device}")

    model_sci = AutoModelForSequenceClassification.from_pretrained(path_sci).to(device)
    model_deb = AutoModelForSequenceClassification.from_pretrained(path_deb).to(device)
    
    tok_sci = AutoTokenizer.from_pretrained(path_sci)
    tok_deb = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    
    # 3. Load Associativity Map (Needed to recreate Phase 5 inputs)
    try:
        with open(Config.ASSOCIATIVITY_MAP_PATH, "r") as f:
            assoc_map = json.load(f)
    except FileNotFoundError:
        print("Warning: Associativity Map not found. Context injection will be skipped (DeBERTa will see raw text).")
        assoc_map = {}

    preds_final = []
    true_labels = dataset["label"]
    disagreements = []

    print("Running Cross-Verification Inference...")
    model_sci.eval()
    model_deb.eval()
    
    with torch.no_grad():
        for i, item in enumerate(dataset):
            text = item["text"]
            true_label = item["label"]
            
            # --- MODEL 1: SciBERT (Standard Input) ---
            in_sci = tok_sci(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            logits_sci = model_sci(**in_sci).logits
            # SOFTMAX: Converts logits to Probabilities (Confidence Score)
            probs_sci = F.softmax(logits_sci, dim=1) 
            
            # --- MODEL 2: Context-DeBERTa (Injected Input) ---
            # Re-inject the context clues exactly like we did in Phase 5
            words = set(text.lower().split())
            hints = []
            for word in words:
                if word in assoc_map:
                    entry = assoc_map[word]
                    syn_str = ",".join(entry['synonyms'])
                    hints.append(f"{word} -> {entry['domain']} (Syns: {syn_str})")
            
            if hints:
                context_str = " | ".join(hints[:3])
                text_injected = f"{text} [SEP] [HINT] {context_str}"
            else:
                text_injected = text
                
            in_deb = tok_deb(text_injected, return_tensors="pt", truncation=True, max_length=512).to(device)
            logits_deb = model_deb(**in_deb).logits
            probs_deb = F.softmax(logits_deb, dim=1)
            
            # --- CROSS-VERIFICATION (Weighted Fusion) ---
            # We give SciBERT 60% weight because it had higher standalone accuracy (98.08%)
            # We give DeBERTa 40% weight
            weighted_probs = (0.6 * probs_sci) + (0.4 * probs_deb)
            
            # The final prediction is the class with the highest combined confidence
            final_pred = torch.argmax(weighted_probs).item()
            preds_final.append(final_pred)
            
            # --- DISAGREEMENT CHECK ---
            pred_sci = torch.argmax(probs_sci).item()
            pred_deb = torch.argmax(probs_deb).item()
            
            if pred_sci != pred_deb:
                # Log cases where models fought
                disagreements.append({
                    "text_snippet": text[:100],
                    "sci_pred": pred_sci,
                    "sci_conf": f"{probs_sci.max().item():.2f}",
                    "deb_pred": pred_deb,
                    "deb_conf": f"{probs_deb.max().item():.2f}",
                    "final_decision": final_pred,
                    "correct": final_pred == true_label
                })

    # 4. Results
    acc = accuracy_score(true_labels, preds_final)
    print(f"\n>>> FINAL SYNERGY ACCURACY: {acc:.4f}")
    
    # 5. Save Disagreement Report (Gold mine for your report analysis)
    if disagreements:
        df_dis = pd.DataFrame(disagreements)
        df_dis.to_csv("phase6_disagreements.csv", index=False)
        print(f"Saved {len(disagreements)} disagreements to phase6_disagreements.csv")

if __name__ == "__main__":
    run_synergy_ensemble()
