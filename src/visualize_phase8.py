import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from src.config import Config
import os

def visualize_fair_duel():
    print("--- GENERATING PHASE 8 VISUALIZATIONS (Debug Mode) ---")
    
    # 1. Load Test Data
    df = pd.read_csv(Config.DATA_PATH)
    # We load via pandas first to easily inspect unique labels
    dataset = load_dataset("csv", data_files=Config.DATA_PATH)["train"].train_test_split(test_size=0.2, seed=42)["test"]
    texts = dataset["text"]
    labels = dataset["label"]
    
    # --- DEBUG: PRINT UNIQUE LABELS ---
    unique_labels = set(labels)
    print(f"\nDEBUG: Found these unique labels in the dataset: {unique_labels}")
    print(f"DEBUG: Type of first label: {type(labels[0])}\n")
    # ----------------------------------

    # 2. Define Paths
    models_config = [
        {"name": "SciBERT", "path": "./models/phase8_scibert_body"},
        {"name": "BERT-Large", "path": "./models/phase8_bert-large_body"}
    ]
    
    plt.figure(figsize=(20, 9))
    
    # 3. UNIVERSAL LABEL MAPPER
    def get_clean_label(val):
        # Case A: It's already the correct string
        s_val = str(val).strip()
        if s_val in ["Neuroscience", "Bioinformatics", "Materials Science"]:
            return s_val
            
        # Case B: It's a number (0, 1, 2) or string number ("0", "1.0")
        try:
            idx = int(float(s_val))
            mapping = {0: "Neuroscience", 1: "Bioinformatics", 2: "Materials Science"}
            return mapping.get(idx, "Unknown")
        except:
            return "Unknown"

    # Map labels once
    plot_labels = [get_clean_label(l) for l in labels]
    
    # Check if mapping worked
    if len(set(plot_labels)) <= 1:
        print("WARNING: Mapping failed (all labels look the same). Plotting raw labels instead.")
        plot_labels = [str(l) for l in labels]

    for i, config in enumerate(models_config):
        name = config["name"]
        path = config["path"]
        
        print(f"\nProcessing {name}...")
        if not os.path.exists(path):
            print(f"CRITICAL WARNING: Path {path} not found.")
            continue
            
        try:
            model = SentenceTransformer(path)
            
            # Encode
            embeddings = model.encode(texts, show_progress_bar=True)
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto")
            X_2d = tsne.fit_transform(embeddings)
            
            # Plot
            plt.subplot(1, 2, i+1)
            
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
                palette="bright", # Forces distinct colors
                s=100, 
                edgecolor="k", 
                alpha=0.8
            )
            
            score = "99.52%" if name == "SciBERT" else "98.56%"
            plt.title(f"{name} Latent Space\nAccuracy: {score}", fontsize=16, fontweight='bold')
            plt.legend(title="Domain", loc="upper right")
            plt.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error processing {name}: {e}")

    plt.tight_layout()
    output_file = "phase8_clash_of_titans.png"
    plt.savefig(output_file)
    print(f"\nVICTORY! Visualization saved to {output_file}")

if __name__ == "__main__":
    visualize_fair_duel()