import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from datasets import Dataset
import sys

# Path hack for config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import Config
except ImportError:
    class Config:
        DATA_PATH = "data/raw/scientific_abstracts_dataset.csv"
        SEED = 42

OUTPUT_DIR = "results/figures"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def generate_comparison():
    print("--- Generating Phase 6 Comparison Visuals ---")
    
    # 1. Load Data
    print("Loading Data...")
    df = pd.read_csv(Config.DATA_PATH)
    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])
    
    # Create Test Set
    dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=Config.SEED)["test"]
    
    # 2. Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    path_sci = "models/phase4_shootout/scibert_scivocab_uncased"
    if not os.path.exists(path_sci): path_sci = "allenai/scibert_scivocab_uncased"
    
    path_deb = "models/deberta_context_aware"
    
    print("Loading SciBERT...")
    model_sci = AutoModelForSequenceClassification.from_pretrained(path_sci).to(device)
    tok_sci = AutoTokenizer.from_pretrained(path_sci)

    print("Loading BERT...")
    path_bert = "models/phase4_shootout/bert-base-uncased"
    if not os.path.exists(path_bert): path_bert = "google-bert/bert-base-uncased"
    model_bert = AutoModelForSequenceClassification.from_pretrained(path_bert).to(device)
    tok_bert = AutoTokenizer.from_pretrained(path_bert)
    
    print("Loading DeBERTa Context...")
    model_deb = AutoModelForSequenceClassification.from_pretrained(path_deb).to(device)
    tok_deb = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
    
    # Load Associativity Map
    with open("data/associativity_map.json", "r") as f:
        assoc_map = json.load(f)
        
    # 3. Inference Loop
    print("Running Inference...")
    
    preds_sci = []
    preds_bert = []
    preds_deb = []
    preds_ens = []
    
    embeddings_sci = []
    embeddings_bert = []
    embeddings_deb = []
    
    true_labels = dataset["label"]
    
    model_sci.eval()
    model_bert.eval()
    model_deb.eval()
    
    # Helper for DeBERTa Context Injection
    def get_context_text(text):
        words = set(text.lower().split())
        hints = []
        for word in words:
            if word in assoc_map:
                entry = assoc_map[word]
                syn_str = ",".join(entry['synonyms'])
                hints.append(f"{word} -> {entry['domain']} (Syns: {syn_str})")
        
        if hints:
            context_str = " | ".join(hints[:3])
            return f"{text} [SEP] [HINT] {context_str}"
        return text

    with torch.no_grad():
        for i, item in enumerate(dataset):
            text = item["text"]
            
            # --- SciBERT ---
            # SciBERT Mapping: 0=Neuro, 1=Mat, 2=Bio
            # Common Mapping (Alpha): 0=Bio, 1=Mat, 2=Neuro
            # We need to permute: [P_neuro, P_mat, P_bio] -> [P_bio, P_mat, P_neuro]
            
            in_sci = tok_sci(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out_sci = model_sci(**in_sci, output_hidden_states=True)
            logits_sci = out_sci.logits
            probs_sci_raw = F.softmax(logits_sci, dim=1)
            
            # Permute to match Common Schema (Bio, Mat, Neuro)
            # SciBERT: [0, 1, 2] -> [2, 1, 0]
            probs_sci = torch.index_select(probs_sci_raw, 1, torch.tensor([2, 1, 0]).to(device))
            
            preds_sci.append(torch.argmax(probs_sci).item())
            
            # Capture Embedding (CLS)
            emb_sci = out_sci.hidden_states[-1][:, 0, :].cpu().numpy()[0]
            embeddings_sci.append(emb_sci)

            # --- BERT ---
            # BERT Mapping is same as SciBERT
            in_bert = tok_bert(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out_bert = model_bert(**in_bert, output_hidden_states=True)
            logits_bert = out_bert.logits
            probs_bert_raw = F.softmax(logits_bert, dim=1)
            
            # Permute
            probs_bert = torch.index_select(probs_bert_raw, 1, torch.tensor([2, 1, 0]).to(device))
            preds_bert.append(torch.argmax(probs_bert).item())
            
            # Capture Embedding (CLS)
            emb_bert = out_bert.hidden_states[-1][:, 0, :].cpu().numpy()[0]
            embeddings_bert.append(emb_bert)
            
            # --- DeBERTa ---
            text_inj = get_context_text(text)
            in_deb = tok_deb(text_inj, return_tensors="pt", truncation=True, max_length=512).to(device)
            out_deb = model_deb(**in_deb, output_hidden_states=True)
            logits_deb = out_deb.logits
            probs_deb = F.softmax(logits_deb, dim=1)
            preds_deb.append(torch.argmax(probs_deb).item())
            
            # Capture Embedding (CLS)
            emb_deb = out_deb.hidden_states[-1][:, 0, :].cpu().numpy()[0]
            embeddings_deb.append(emb_deb)
            
            # --- Ensemble (SciBERT + DeBERTa) ---
            weighted_probs = (0.6 * probs_sci) + (0.4 * probs_deb)
            preds_ens.append(torch.argmax(weighted_probs).item())
            
    # 4. Metrics
    acc_sci = accuracy_score(true_labels, preds_sci)
    acc_bert = accuracy_score(true_labels, preds_bert)
    acc_deb = accuracy_score(true_labels, preds_deb)
    acc_ens = accuracy_score(true_labels, preds_ens)
    
    print(f"SciBERT Accuracy: {acc_sci:.4f}")
    print(f"BERT Accuracy: {acc_bert:.4f}")
    print(f"DeBERTa Accuracy: {acc_deb:.4f}")
    print(f"Ensemble Accuracy: {acc_ens:.4f}")
    
    # 5. Plot Performance
    data = [
        {"Model": "SciBERT", "Accuracy": acc_sci},
        {"Model": "BERT", "Accuracy": acc_bert},
        {"Model": "DeBERTa + Context", "Accuracy": acc_deb},
        {"Model": "Synergy Ensemble", "Accuracy": acc_ens}
    ]
    df_res = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_res, x="Model", y="Accuracy", palette="viridis")
    plt.title("Phase 6: Synergy Ensemble Performance", fontsize=15, fontweight='bold')
    plt.ylim(0.90, 1.01)
    for i in ax.containers: ax.bar_label(i, fmt='%.4f', padding=3, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase6_comparison_performance.png")
    print("Saved performance comparison.")
    
    # 6. Plot Clusters
    print("Generating t-SNE plots...")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    X_sci = tsne.fit_transform(np.array(embeddings_sci))
    X_bert = tsne.fit_transform(np.array(embeddings_bert))
    X_deb = tsne.fit_transform(np.array(embeddings_deb))
    
    plt.figure(figsize=(24, 6))
    
    # Plot 1: SciBERT
    plt.subplot(1, 4, 1)
    sns.scatterplot(x=X_sci[:,0], y=X_sci[:,1], hue=preds_sci, palette="tab10", s=60, legend=False)
    plt.title(f"SciBERT\n({acc_sci:.1%})", fontweight='bold')
    plt.axis('off')

    # Plot 2: BERT
    plt.subplot(1, 4, 2)
    sns.scatterplot(x=X_bert[:,0], y=X_bert[:,1], hue=preds_bert, palette="tab10", s=60, legend=False)
    plt.title(f"BERT\n({acc_bert:.1%})", fontweight='bold')
    plt.axis('off')
    
    # Plot 3: DeBERTa
    plt.subplot(1, 4, 3)
    sns.scatterplot(x=X_deb[:,0], y=X_deb[:,1], hue=preds_deb, palette="tab10", s=60, legend=False)
    plt.title(f"DeBERTa + Context\n({acc_deb:.1%})", fontweight='bold')
    plt.axis('off')
    
    # Plot 4: Ensemble
    plt.subplot(1, 4, 4)
    sns.scatterplot(x=X_deb[:,0], y=X_deb[:,1], hue=preds_ens, palette="tab10", s=60, legend=False)
    plt.title(f"Ensemble (Sci+Deb)\n({acc_ens:.1%})", fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase6_comparison_clusters.png")
    print("Saved cluster comparison.")

if __name__ == "__main__":
    generate_comparison()
