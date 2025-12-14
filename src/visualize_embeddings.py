import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from src.config import Config

# Models to compare (Adjust paths based on Shootout results)
MODELS_TO_PLOT = {
    "BERT (Baseline)": f"{Config.PHASE4_DIR}/bert-base-uncased",
    "DeBERTa (Challenger)": f"{Config.PHASE4_DIR}/deberta-v3-base"
}

def get_embeddings(model_path, dataset):
    print(f"Generating vectors for {model_path}...")
    # Note: We load AutoModel (encoder only), not Classification Head
    model = AutoModel.from_pretrained(model_path).to(Config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    embeddings = []
    labels = []
    
    # Process in small batches
    batch_size = 16
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=512, return_tensors="pt").to(Config.DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token (first token) as the sentence vector
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        embeddings.extend(cls_emb)
        labels.extend(batch["label"])
        
    return embeddings, labels

def run_visualization():
    # Load Test Data
    # Load Test Data
    dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train").class_encode_column("label").train_test_split(test_size=0.2, seed=42)["test"]
    
    plt.figure(figsize=(14, 6))
    
    for idx, (name, path) in enumerate(MODELS_TO_PLOT.items()):
        try:
            embs, labels = get_embeddings(path, dataset)
            embs = np.array(embs)
            
            # Adjust perplexity for small datasets
            n_samples = len(embs)
            perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
            
            # Run T-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
            embs_2d = tsne.fit_transform(embs)
            
            # Plot
            df = pd.DataFrame(embs_2d, columns=["x", "y"])
            df["label"] = labels
            
            plt.subplot(1, 2, idx + 1)
            sns.scatterplot(data=df, x="x", y="y", hue="label", palette="tab10", s=15, alpha=0.7)
            plt.title(f"{name} Cluster Separation")
            plt.axis("off")
        except Exception as e:
            print(f"Skipping visualization for {name}: {e}")
        
    plt.tight_layout()
    plt.savefig("phase4_cluster_comparison.png")
    print("Visualization saved to phase4_cluster_comparison.png")

if __name__ == "__main__":
    run_visualization()
