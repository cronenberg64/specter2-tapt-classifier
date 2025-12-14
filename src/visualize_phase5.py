import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import sys
import os
# Add project root to sys.path to allow running script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import Config
import json
import os

def get_embeddings(model_path, dataset, assoc_map=None):
    print(f"Generating vectors for {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model body only (no classification head)
    try:
        model = AutoModel.from_pretrained(model_path).to(device)
    except:
        # Fallback if it's a sequence classification model save
        from transformers import AutoModelForSequenceClassification
        full_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = full_model.base_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for item in dataset:
            text = item["text"]
            
            # Inject context if map provided (for DeBERTa)
            if assoc_map:
                words = set(text.lower().split())
                hints = []
                for word in words:
                    if word in assoc_map:
                        entry = assoc_map[word]
                        syn_str = ",".join(entry['synonyms'])
                        hints.append(f"{word} -> {entry['domain']} (Syns: {syn_str})")
                if hints:
                    context_str = " | ".join(hints[:3])
                    text = f"{text} [SEP] [HINT] {context_str}"

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            # Use CLS token (first token)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            embeddings.append(cls_emb)
            labels.append(item["label"])
            
    return np.array(embeddings), labels

def run_visualization():
    # Load Test Data
    dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train")
    dataset = dataset.class_encode_column("label")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)["test"]
    
    # Load Map
    with open(Config.ASSOCIATIVITY_MAP_PATH, "r") as f:
        assoc_map = json.load(f)

    models_to_plot = {
        "SciBERT (Phase 4)": os.path.join(Config.PHASE4_DIR, "scibert_scivocab_uncased"),
        "Context-DeBERTa (Phase 5)": os.path.join(Config.PROJECT_ROOT, "models/deberta_context_aware")
    }
    
    plt.figure(figsize=(14, 6))
    
    for idx, (name, path) in enumerate(models_to_plot.items()):
        try:
            # Pass assoc_map only for DeBERTa
            amap = assoc_map if "DeBERTa" in name else None
            
            print(f"Processing {name}...")
            print(f"Path: {path}")
            print(f"Assoc Map: {'Loaded' if amap else 'None'}")
            
            embs, labels = get_embeddings(path, dataset, amap)
            
            # Adjust perplexity for small datasets
            n_samples = len(embs)
            perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
            embs_2d = tsne.fit_transform(embs)
            
            plt.subplot(1, 2, idx+1)
            sns.scatterplot(x=embs_2d[:,0], y=embs_2d[:,1], hue=labels, palette="deep", s=60)
            plt.title(f"{name} Clusters")
            
        except Exception as e:
            print(f"Skipping visualization for {name}: {e}")
            
    plt.tight_layout()
    plt.savefig("phase5_cluster_comparison.png")
    print("Visualization saved to phase5_cluster_comparison.png")

if __name__ == "__main__":
    run_visualization()
