import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from src.config import Config

# Models to compare (Adjust paths based on Shootout results)
# We assume the shootout saves them in Config.PHASE4_DIR
MODELS_TO_PLOT = {
    "BERT (Baseline)": f"{Config.PHASE4_DIR}/bert-base-uncased",
    "RoBERTa-Large (Challenger)": f"{Config.PHASE4_DIR}/roberta-large"
}

def get_embeddings(model_path, dataset):
    print(f"Generating vectors for {model_path}...")
    
    # Check if path exists
    if not os.path.exists(model_path):
        print(f"WARNING: Model path {model_path} not found. Skipping.")
        return None, None

    # Note: We load AutoModel (encoder only), not Classification Head
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    embeddings = []
    labels = []
    
    # Process in small batches
    batch_size = 16
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token (first token) as the sentence vector
            # For DeBERTa, it's also the first token usually, but let's be safe.
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        embeddings.extend(cls_emb)
        labels.extend(batch["label"])
        
    return embeddings, labels

def run_visualization():
    # Load Test Data
    # We need to replicate the split logic to get the same test set
    dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = dataset["test"]
    
    plt.figure(figsize=(14, 6))
    
    valid_plots = 0
    for idx, (name, path) in enumerate(MODELS_TO_PLOT.items()):
        embs, labels = get_embeddings(path, test_dataset)
        
        if embs is None:
            continue
            
        valid_plots += 1
        
        # Run T-SNE
        print(f"Running t-SNE for {name}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
        embs_2d = tsne.fit_transform(embs)
        
        # Plot
        df = pd.DataFrame(embs_2d, columns=["x", "y"])
        df["label"] = labels
        
        plt.subplot(1, 2, idx + 1)
        sns.scatterplot(data=df, x="x", y="y", hue="label", palette="tab10", s=15, alpha=0.7)
        plt.title(f"{name} Cluster Separation")
        plt.axis("off")
        
    if valid_plots > 0:
        if not os.path.exists("results"):
            os.makedirs("results")
        output_path = "results/phase4_cluster_comparison.png"
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        print("No valid models found to plot.")

if __name__ == "__main__":
    run_visualization()
