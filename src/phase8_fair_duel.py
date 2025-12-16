import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from src.config import Config

def visualize_fair_duel():
    print("--- GENERATING PHASE 8 VISUALIZATIONS ---")
    
    # Load Test Data
    df = pd.read_csv(Config.DATA_PATH)
    # Ensure strict test split
    dataset = load_dataset("csv", data_files=Config.DATA_PATH)["train"].train_test_split(test_size=0.2, seed=42)["test"]
    texts = dataset["text"]
    labels = dataset["label"]
    
    # Load the Trained Bodies
    models_to_plot = ["SciBERT", "BERT-Large"]
    paths = ["./models/phase8_scibert_body", "./models/phase8_bert-large_body"]
    
    plt.figure(figsize=(20, 8))
    
    for i, (name, path) in enumerate(zip(models_to_plot, paths)):
        print(f"Loading {name} from {path}...")
        
        try:
            model = SentenceTransformer(path)
            # Encode
            print(f"Encoding {len(texts)} samples...")
            embeddings = model.encode(texts)
            
            # t-SNE
            print(f"Running t-SNE for {name}...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto")
            X_2d = tsne.fit_transform(embeddings)
            
            # Plot
            plt.subplot(1, 2, i+1)
            sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=labels, palette="deep", s=60, edgecolor="k")
            plt.title(f"{name} (SetFit)\nAccuracy: {0.9952 if name=='SciBERT' else 0.9856}", fontsize=16)
            plt.legend(title="Class")
            
        except Exception as e:
            print(f"Skipping {name}: {e}")

    plt.tight_layout()
    plt.savefig("phase8_clash_of_titans.png")
    print("Saved comparison plot to phase8_clash_of_titans.png")

if __name__ == "__main__":
    visualize_fair_duel()