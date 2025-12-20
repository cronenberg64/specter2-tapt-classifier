import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.manifold import TSNE

# Config
LEADERBOARD = "results/phase4_leaderboard.json"
ENSEMBLE = "results/phase4_ensemble_result.json"
OUTPUT_DIR = "results/figures"

def plot_leaderboard():
    if not os.path.exists(LEADERBOARD): return
    with open(LEADERBOARD, 'r') as f: data = json.load(f)
    if not data: return
    
    df = pd.DataFrame(data)
    df = df.sort_values("accuracy", ascending=False)
    
    plt.figure(figsize=(10,6))
    ax = sns.barplot(data=df, x="model", y="accuracy", palette="viridis")
    plt.title("Phase 4 Best Leaderboard", fontsize=15, fontweight='bold')
    plt.ylim(0.9, 1.0)
    plt.xticks(rotation=15)
    for i in ax.containers: ax.bar_label(i, fmt='%.3f', padding=3, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase4_shootout_leaderboard.png")
    print("Saved leaderboard.")

def plot_ensemble_gain():
    if not os.path.exists(ENSEMBLE): return
    with open(ENSEMBLE, 'r') as f: ens = json.load(f)
    
    if not os.path.exists(LEADERBOARD): return
    with open(LEADERBOARD, 'r') as f: data = json.load(f)
    
    # Collect all metrics
    metrics = []
    for m in data:
        name = m["model"].split("/")[-1]
        metrics.append({"Model": name, "Accuracy": m["accuracy"], "Type": "Single"})
        
    # Construct descriptive label
    components = [m["model"].split("/")[-1].split("-")[0] for m in ens.get("top_models", [])]
    label = f"Ensemble ({' + '.join(components)})" if components else "Ensemble (Soft Vote)"
    
    metrics.append({"Model": label, "Accuracy": ens["ensemble_accuracy"], "Type": "Ensemble"})
    
    df = pd.DataFrame(metrics)
    df = df.sort_values("Accuracy", ascending=True)
    
    plt.figure(figsize=(10,6))
    # Highlight Ensemble with distinct color
    ax = sns.barplot(data=df, x="Model", y="Accuracy", hue="Type", palette={"Single": "#95a5a6", "Ensemble": "#e74c3c"}, dodge=False)
    plt.title("Ensemble Impact Analysis (Full Comparison)", fontsize=15, fontweight='bold')
    
    # Dynamic Y-lim
    min_acc = df["Accuracy"].min()
    max_acc = df["Accuracy"].max()
    plt.ylim(min_acc * 0.98, max_acc * 1.002)
    
    for i in ax.containers: ax.bar_label(i, fmt='%.4f', padding=3, fontweight='bold')
    plt.legend(title="Model Type")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase4_ensemble_gain.png")
    print("Saved ensemble gain.")

def plot_clusters():
    # Load Top 2 models
    if not os.path.exists(ENSEMBLE): return
    with open(ENSEMBLE, 'r') as f: ens = json.load(f)
    top2 = ens["top_models"]
    
    from datasets import load_dataset
    # Use Config path logic but hardcode for simplicity as we know it
    DATA_PATH = "data/raw/scientific_abstracts_dataset.csv"
    dataset = load_dataset("csv", data_files=DATA_PATH, split="train").train_test_split(test_size=0.1, seed=42)["test"]
    texts = dataset["text"]
    labels = dataset["label"]
    
    plt.figure(figsize=(16, 7))
    
    for i, m_info in enumerate(top2):
        name = m_info["model"]
        path = m_info["path"]
        print(f"Processing clusters for {name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path).cuda() if torch.cuda.is_available() else AutoModel.from_pretrained(path)
        model.eval()
        
        embeddings = []
        batch_size = 16
        for j in range(0, len(texts), batch_size):
            batch = texts[j:j+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            if torch.cuda.is_available(): inputs = {k:v.cuda() for k,v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(emb)
        
        embeddings = np.vstack(embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = tsne.fit_transform(embeddings)
        
        plt.subplot(1, 2, i+1)
        sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=labels, palette="tab10", s=60, alpha=0.7)
        plt.title(f"{name.split('/')[-1]} Latent Space", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase4_cluster_comparison.png")
    print("Saved clusters.")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    plot_leaderboard()
    plot_ensemble_gain()
    plot_clusters()
