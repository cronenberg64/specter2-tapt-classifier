import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import Config
except ImportError:
    class Config:
        DATA_PATH = "data/raw/scientific_abstracts_dataset.csv"

def visualize_finetuning_comparison():
    print("--- GENERATING FINE-TUNING COMPARISON VISUALIZATION ---")
    
    # Load Test Data
    dataset = load_dataset("csv", data_files=Config.DATA_PATH)["train"].train_test_split(test_size=0.2, seed=42)["test"]
    texts = dataset["text"]
    labels = dataset["label"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Fine-tuned models from Phase 4/5
    models_to_plot = ["SciBERT", "BERT-base", "RoBERTa-Large", "DeBERTa + Context"]
    paths = [
        "./models/phase4_shootout/scibert_scivocab_uncased",
        "./models/phase4_shootout/bert-base-uncased",
        "./models/phase4_shootout/roberta-large",
        "./models/deberta_context_aware"
    ]
    # Hardcoded scores
    scores = [0.9712, 0.9808, 0.96, 0.9615]  # SciBERT, BERT-base, RoBERTa, DeBERTa
    
    plt.figure(figsize=(28, 7))
    
    for i, (name, path, score) in enumerate(zip(models_to_plot, paths, scores)):
        print(f"Loading {name} from {path}...")
        
        if not os.path.exists(path):
            print(f"WARNING: {path} not found. Skipping.")
            continue
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
        except:
            # Fallback to base tokenizer if saved model doesn't have it
            base_name = "roberta-large" if "roberta" in path.lower() else path
            tokenizer = AutoTokenizer.from_pretrained(base_name)
        
        try:
            model = AutoModel.from_pretrained(path).to(device)
            model.eval()
            
            # Encode (Get CLS embeddings)
            print(f"Encoding {len(texts)} samples...")
            embeddings = []
            with torch.no_grad():
                for text in texts:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                    out = model(**inputs)
                    emb = out.last_hidden_state[:, 0, :].cpu().numpy()[0]
                    embeddings.append(emb)
            
            embeddings = np.array(embeddings)
            
            # t-SNE
            print(f"Running t-SNE for {name}...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto")
            X_2d = tsne.fit_transform(embeddings)
            
            # Plot
            plt.subplot(1, 4, i+1)
            sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=labels, palette="deep", s=60, edgecolor="k")
            plt.title(f"{name} (Fine-tuned)\nAccuracy: {score:.2%}", fontsize=16)
            plt.legend(title="Class")
            
        except Exception as e:
            print(f"Skipping {name}: {e}")

    plt.suptitle("Fine-tuning Comparison (Standard Training)", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("results/figures/finetuning_comparison.png", bbox_inches='tight', dpi=150)
    print("Saved comparison plot to results/figures/finetuning_comparison.png")

if __name__ == "__main__":
    visualize_finetuning_comparison()
