import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
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

OUTPUT_DIR = "results/figures"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def get_clean_label(val):
    """Universal label mapper."""
    s_val = str(val).strip()
    if s_val in ["Neuroscience", "Bioinformatics", "Materials Science"]:
        return s_val
    try:
        idx = int(float(s_val))
        mapping = {0: "Neuroscience", 1: "Bioinformatics", 2: "Materials Science"}
        return mapping.get(idx, "Unknown")
    except:
        return "Unknown"

def get_bert_base_embeddings(texts, device):
    """Get embeddings from fine-tuned BERT-base model."""
    path = "models/phase4_shootout/bert-base-uncased"
    if not os.path.exists(path):
        path = "google-bert/bert-base-uncased"
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).to(device)
    model.eval()
    
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out = model(**inputs)
            # CLS token embedding
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()[0]
            embeddings.append(emb)
    
    return np.array(embeddings)

def visualize_phase8_with_bert_base():
    print("--- GENERATING PHASE 8 VISUALIZATIONS (with BERT-base) ---")
    
    # 1. Load Test Data
    dataset = load_dataset("csv", data_files=Config.DATA_PATH)["train"].train_test_split(test_size=0.2, seed=42)["test"]
    texts = dataset["text"]
    labels = dataset["label"]
    
    # Map labels
    plot_labels = [get_clean_label(l) for l in labels]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2. Define Models (All SetFit trained for fair comparison)
    models_config = [
        {"name": "SciBERT (SetFit)", "path": "./models/phase8_scibert_body", "type": "sentence_transformer"},
        {"name": "BERT-Large (SetFit)", "path": "./models/phase8_bert-large_body", "type": "sentence_transformer"},
        {"name": "BERT-base (SetFit)", "path": "./models/phase8_bert-base_body", "type": "sentence_transformer"}
    ]
    
    plt.figure(figsize=(24, 8))
    
    for i, config in enumerate(models_config):
        name = config["name"]
        path = config["path"]
        model_type = config["type"]
        
        print(f"\nProcessing {name}...")
        
        if not os.path.exists(path):
            print(f"WARNING: Path {path} not found. Skipping.")
            continue
        try:
            model = SentenceTransformer(path)
            embeddings = model.encode(texts, show_progress_bar=True)
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto")
        X_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.subplot(1, 3, i+1)
        
        plot_df = pd.DataFrame({
            'x': X_2d[:,0],
            'y': X_2d[:,1],
            'Field': plot_labels
        })
        
        sns.scatterplot(
            data=plot_df,
            x='x', y='y', 
            hue='Field', 
            style='Field',
            palette="bright",
            s=100, 
            edgecolor="k", 
            alpha=0.8
        )
        
        # Hardcoded scores from Phase 8 SetFit training
        if "SciBERT" in name:
            score = "99.52%"
        elif "Large" in name:
            score = "98.56%"
        else:  # BERT-base SetFit
            score = "94.23%"
            
        plt.title(f"{name}\nAccuracy: {score}", fontsize=14, fontweight='bold')
        plt.legend(title="Domain", loc="upper right", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.xlabel("")
        plt.ylabel("")
    
    plt.suptitle("Phase 8: Clash of the Titans (with BERT-base)", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = f"{OUTPUT_DIR}/phase8_final_cluster.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"\nSaved visualization to {output_file}")

if __name__ == "__main__":
    visualize_phase8_with_bert_base()
