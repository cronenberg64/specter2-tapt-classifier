import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification

# Config
SPECTER_RESULTS = "results/phase3_specter_results.json"
MODERNBERT_RESULTS = "results/phase3_modernbert_results.json"
DATA_PATH = "data/raw/scientific_abstracts_dataset.csv"
OUTPUT_DIR = "results/figures"

# Models to compare
from transformers import AutoModel, AutoTokenizer
import torch

# Models to compare
# Pointing to specific checkpoints where weights exist
MODELS = {
    "Specter Full Fine-Tune": "./models/specter_full_finetune/checkpoint-520",
    "ModernBERT": "./models/modernbert_classifier/checkpoint-520"
}

def get_embeddings_transformers(model_path, texts):
    print(f"Extracting embeddings using Transformers for {model_path}...")
    try:
        # 1. Try loading tokenizer from checkpoint
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        print("Tokenizer not found in checkpoint. Falling back to base model...")
        # 2. Fallback to base model tokenizers
        if "specter" in model_path.lower():
            tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        elif "modernbert" in model_path.lower():
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", trust_remote_code=True)
        else:
            return None # Fail safe

    try:
        # Load model from checkpoint
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Transformers model load failed: {e}")
        return None # Return None to signal failure

    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        
    embeddings = []
    batch_size = 16
    
    # Debug: Check first batch
    first_batch = True
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                if first_batch:
                    print(f"DEBUG: Batch shape: {batch_embeddings.shape}, Mean: {np.mean(batch_embeddings)}")
                    first_batch = False
                    
                embeddings.append(batch_embeddings)
        except Exception as e:
            print(f"Batch processing failed: {e}")
            continue
            
    if not embeddings:
        return None
            
    return np.vstack(embeddings)

def generate_graphs():
    print("--- Generating Phase 3 Graphs ---")
    
    # Load Data
    try:
        with open(SPECTER_RESULTS, 'r') as f:
            specter_data = json.load(f)
        with open(MODERNBERT_RESULTS, 'r') as f:
            modernbert_data = json.load(f)
    except FileNotFoundError:
        print("Results file not found. Skipping benchmark graphs.")
        return
        
    # Prepare DataFrame
    metrics = []
    
    # Specter
    if "Full_FineTune" in specter_data:
        s = specter_data["Full_FineTune"]
        metrics.append({
            "Model": "Specter Full Fine-Tune",
            "Accuracy": s["eval_accuracy"],
            "F1": s["eval_f1"],
            "Samples/Sec": s["eval_samples_per_second"]
        })
        
    # ModernBERT
    metrics.append({
        "Model": "ModernBERT Classifier",
        "Accuracy": modernbert_data["eval_accuracy"],
        "F1": modernbert_data["eval_f1"],
        "Samples/Sec": modernbert_data["eval_samples_per_second"]
    })
    
    df = pd.DataFrame(metrics)
    
    # 1. Benchmark (Accuracy)
    if not df.empty:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, x="Model", y="Accuracy", palette="viridis")
        plt.title("Phase 3 Benchmark: Accuracy Leaderboard", fontsize=15, fontweight='bold')
        plt.ylim(0.9, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        for i in ax.containers:
            ax.bar_label(i, fmt='%.3f', padding=3, fontweight='bold', fontsize=12)
            
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/phase3_beat_bert_benchmark.png")
        print(f"Saved: {OUTPUT_DIR}/phase3_beat_bert_benchmark.png")
        
        # 2. Training Efficiency (Samples/Sec)
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, x="Model", y="Samples/Sec", palette="plasma")
        plt.title("Inference Throughput Analysis (Samples/Sec)", fontsize=15, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        for i in ax.containers:
            ax.bar_label(i, fmt='%.1f', padding=3, fontweight='bold', fontsize=12)
            
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/phase3_training_efficiency.png")
        print(f"Saved: {OUTPUT_DIR}/phase3_training_efficiency.png")

def generate_clusters():
    print("--- Generating Phase 3 Clusters ---")
    # Load Data
    dataset = load_dataset("csv", data_files=DATA_PATH)["train"].train_test_split(test_size=0.1, seed=42)["test"]
    texts = dataset["text"]
    labels = dataset["label"]
    
    # Collect valid embeddings first
    valid_results = []
    for name, path in MODELS.items():
        print(f"Processing {name} at {path}...")
        try:
            if not os.path.exists(path): 
                print(f"Path not found: {path} - Skipping")
                continue
                
            try:
                model = SentenceTransformer(path)
                embeddings = model.encode(texts, show_progress_bar=True)
            except Exception:
                embeddings = get_embeddings_transformers(path, texts)
                
            if embeddings is not None:
                valid_results.append((name, embeddings))
        except Exception:
            continue

    if not valid_results:
        return

    # Dynamic Layout
    n_plots = len(valid_results)
    plt.figure(figsize=(10 * n_plots, 8))
    
    for i, (name, embeddings) in enumerate(valid_results):
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = tsne.fit_transform(embeddings)
        
        plt.subplot(1, n_plots, i+1)
        sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=labels, palette="tab10", s=60, alpha=0.7)
        plt.title(f"{name} Latent Space", fontsize=16, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase3_latent_clusters.png")
    print(f"Saved: {OUTPUT_DIR}/phase3_latent_clusters.png")
    plt.savefig(f"{OUTPUT_DIR}/phase3_latent_clusters.png")
    print(f"Saved: {OUTPUT_DIR}/phase3_latent_clusters.png")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    generate_graphs()
    generate_clusters()
