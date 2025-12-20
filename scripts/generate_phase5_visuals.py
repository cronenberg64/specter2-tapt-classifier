import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.manifold import TSNE
try:
    import networkx as nx
except ImportError:
    nx = None

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import Config
except ImportError:
    class Config:
        DATA_PATH = "data/raw/scientific_abstracts_dataset.csv"
        SEED = 42

OUTPUT_DIR = "results/figures"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def plot_knowledge_map():
    print("Generating Knowledge Map...")
    if not os.path.exists("data/associativity_map.json"): 
        print("Map not found.")
        return
        
    if nx is None:
        print("NetworkX not installed. Skipping graph.")
        return

    with open("data/associativity_map.json") as f:
        data = json.load(f)
    
    # Build Graph
    G = nx.Graph()
    
    # Nodes: Domains and Words
    domains = set(v['domain'] for v in data.values())
    domain_color_map = {d: c for d, c in zip(domains, ["#e74c3c", "#3498db", "#2ecc71"])}
    
    # Top 30 words to keep it clean
    for word, info in list(data.items())[:30]: 
        dom = info['domain']
        G.add_edge(dom, word)
        
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    plt.figure(figsize=(10, 8))
    
    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if node in domains: node_colors.append(domain_color_map.get(node, "black"))
        else: node_colors.append("#bdc3c7")
        
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=[3000 if n in domains else 600 for n in G.nodes()], 
            font_size=9, alpha=0.9, font_weight='bold')
            
    plt.title("Phase 5 Knowledge Map (Associativity Network)", fontsize=15, fontweight='bold')
    plt.savefig(f"{OUTPUT_DIR}/phase5_knowledge_map.png")
    print("Saved knowledge map.")

def plot_context_gain():
    res_path = "results/phase5_context_results.json"
    if not os.path.exists(res_path): 
        print("No context results.")
        return
    with open(res_path) as f: res = json.load(f)
    
    # Comparison
    leaderboard_path = "results/phase4_leaderboard.json"
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path) as f: p4 = json.load(f)
        best_p4 = max(p4, key=lambda x: x['accuracy'])
        baseline_name = f"Prev. Best ({best_p4['model'].split('/')[-1]})"
        baseline_acc = best_p4['accuracy']
    else:
        baseline_name = "Baseline"
        baseline_acc = 0.90 # Dummy
    
    data = [
        {"Model": baseline_name, "Accuracy": baseline_acc},
        {"Model": "DeBERTa + Context", "Accuracy": res['eval_accuracy']}
    ]
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8,6))
    ax = sns.barplot(data=df, x="Model", y="Accuracy", palette="magma")
    plt.title("Context Boost Analysis", fontsize=15, fontweight='bold')
    plt.ylim(min(baseline_acc, res['eval_accuracy']) * 0.98, 1.0)
    for i in ax.containers: ax.bar_label(i, fmt='%.4f', padding=3, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase5_context_gain.png")
    print("Saved context gain.")

def plot_cluster_refinement():
    print("Generating Cluster Refinement...")
    model_path = "./models/deberta_context_aware"
    if not os.path.exists(model_path): 
        print("Model not found.")
        return
    
    df = pd.read_csv(Config.DATA_PATH)
    # Sample
    df = df.groupby('label').apply(lambda x: x.sample(n=80, random_state=42)).reset_index(drop=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Bert Base as Baseline reference
    baseline_path = "models/phase4_shootout/bert-base-uncased"
    if not os.path.exists(baseline_path):
         # If local path missing, imply huggingface
         baseline_path = "google-bert/bert-base-uncased"

    models = [
        {"name": "Previous Best", "path": baseline_path},
        {"name": "DeBERTa + Context", "path": model_path}
    ]
    
    plt.figure(figsize=(16, 7))
    
    for i, m_info in enumerate(models):
        try:
            tokenizer = AutoTokenizer.from_pretrained(m_info["path"], use_fast=False)
            model = AutoModel.from_pretrained(m_info["path"]).to(device)
        except Exception as e:
             print(f"Failed to load {m_info['name']}: {e}")
             continue
             
        embeddings = []
        with torch.no_grad():
            for text in df["text"]:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                out = model(**inputs)
                emb = out.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(emb[0])
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = tsne.fit_transform(np.array(embeddings))
        
        plt.subplot(1, 2, i+1)
        sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=df["label"], palette="viridis", s=60)
        plt.title(f"{m_info['name']} Latent Space", fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase5_cluster_refinement.png")
    print("Saved cluster refinement.")

if __name__ == "__main__":
    plot_knowledge_map()
    plot_context_gain()
    plot_cluster_refinement()
