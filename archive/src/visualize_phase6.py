import sys
import os
# Add project root to sys.path to allow running script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.phase6_ensemble import run_synergy_ensemble # We can reuse the logic or just load the csv if we saved all preds
# Actually, let's just run inference again to get the data for plotting, 
# or better, let's modify this script to import the logic if possible.
# For simplicity, I will re-implement the inference loop here to capture all data needed for plotting.

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import json
from src.config import Config
import numpy as np

def run_phase6_viz():
    print("--- GENERATING PHASE 6 VISUALIZATIONS ---")
    
    # 1. Load Data
    dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train")
    dataset = dataset.class_encode_column("label")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)["test"]
    
    # 2. Load Models
    path_sci = os.path.join(Config.PHASE4_DIR, "scibert_scivocab_uncased")
    path_deb = os.path.join(Config.PROJECT_ROOT, "models/deberta_context_aware")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading SciBERT from: {path_sci}")
    model_sci = AutoModelForSequenceClassification.from_pretrained(path_sci).to(device)
    print(f"Loading DeBERTa from: {path_deb}")
    model_deb = AutoModelForSequenceClassification.from_pretrained(path_deb).to(device)
    
    tok_sci = AutoTokenizer.from_pretrained(path_sci)
    tok_deb = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    
    with open(Config.ASSOCIATIVITY_MAP_PATH, "r") as f:
        assoc_map = json.load(f)

    preds = []
    confs = []
    embeddings_sci = [] # Capture SciBERT embeddings for plotting
    labels = dataset["label"]
    
    model_sci.eval()
    model_deb.eval()
    
    print("Running Ensemble Inference & Extracting Embeddings...")
    with torch.no_grad():
        for i, item in enumerate(dataset):
            text = item["text"]
            
            # SciBERT
            in_sci = tok_sci(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out_sci = model_sci(**in_sci, output_hidden_states=True)
            logits_sci = out_sci.logits
            probs_sci = F.softmax(logits_sci, dim=1)
            
            # Capture CLS embedding (last hidden state, first token)
            # hidden_states is a tuple, last one is the last layer
            cls_emb = out_sci.hidden_states[-1][:, 0, :].cpu().numpy()[0]
            embeddings_sci.append(cls_emb)
            
            # DeBERTa
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
            probs_deb = F.softmax(model_deb(**in_deb).logits, dim=1)
            
            # Ensemble
            weighted_probs = (0.6 * probs_sci) + (0.4 * probs_deb)
            final_pred = torch.argmax(weighted_probs).item()
            final_conf = weighted_probs.max().item()
            
            preds.append(final_pred)
            confs.append(final_conf)

    # --- PLOT 1: Confusion Matrix ---
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Phase 6 Ensemble Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("phase6_confusion_matrix.png")
    print("Saved phase6_confusion_matrix.png")
    
    # --- PLOT 2: Confidence Histogram ---
    correct_confs = [c for p, l, c in zip(preds, labels, confs) if p == l]
    incorrect_confs = [c for p, l, c in zip(preds, labels, confs) if p != l]
    
    plt.figure(figsize=(10, 6))
    plt.hist(correct_confs, bins=20, alpha=0.7, label='Correct Predictions', color='green')
    plt.hist(incorrect_confs, bins=20, alpha=0.7, label='Incorrect Predictions', color='red')
    plt.title('Confidence Distribution: Correct vs Incorrect')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig("phase6_confidence_histogram.png")
    print("Saved phase6_confidence_histogram.png")

    # --- PLOT 3: Ensemble Clusters (t-SNE) ---
    print("Generating t-SNE for Ensemble Clusters...")
    from sklearn.manifold import TSNE
    embs = np.array(embeddings_sci)
    n_samples = len(embs)
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
    embs_2d = tsne.fit_transform(embs)
    
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: True Labels
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=embs_2d[:,0], y=embs_2d[:,1], hue=labels, palette="deep", s=60)
    plt.title("True Labels (Ground Truth)")
    
    # Subplot 2: Ensemble Predictions
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=embs_2d[:,0], y=embs_2d[:,1], hue=preds, palette="deep", s=60)
    plt.title("Ensemble Predictions")
    
    plt.tight_layout()
    plt.savefig("phase6_ensemble_clusters.png")
    print("Saved phase6_ensemble_clusters.png")

if __name__ == "__main__":
    run_phase6_viz()
