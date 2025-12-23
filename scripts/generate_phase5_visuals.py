import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
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

BASELINE_MODEL_DIR = "models/deberta_baseline"

def ensure_baseline_model():
    """Trains a standard DeBERTa baseline if it doesn't exist."""
    if os.path.exists(os.path.join(BASELINE_MODEL_DIR, "config.json")):
        print(f"Baseline model found at {BASELINE_MODEL_DIR}")
        return

    print("--- Training Baseline DeBERTa (No Context) ---")
    try:
        # Load Data
        print("Loading data...")
        df = pd.read_csv(Config.DATA_PATH)
        print(f"Data loaded: {len(df)} rows")
        
        # Encode labels
        df["label"] = LabelEncoder().fit_transform(df["label"])
        
        dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=Config.SEED)
        print("Dataset split created")
        
        model_name = "microsoft/deberta-v3-base"
        print(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
            
        print("Tokenizing dataset...")
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        print("Tokenization complete")
        
        print(f"Loading model {model_name}...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        print("Model loaded")
        
        args = TrainingArguments(
            output_dir=BASELINE_MODEL_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            learning_rate=2e-5,
            evaluation_strategy="no",
            save_strategy="no",
            report_to="none",
            fp16=torch.cuda.is_available()
        )
        
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"]
        )
        
        print("Starting training...")
        trainer.train()
        print("Training complete")
        trainer.save_model(BASELINE_MODEL_DIR)
        tokenizer.save_pretrained(BASELINE_MODEL_DIR)
        print("Baseline model trained and saved.")
    except Exception as e:
        print(f"CRITICAL ERROR in ensure_baseline_model: {e}")
        import traceback
        traceback.print_exc()
        # Clean up empty directory
        if os.path.exists(BASELINE_MODEL_DIR) and not os.listdir(BASELINE_MODEL_DIR):
            os.rmdir(BASELINE_MODEL_DIR)

def get_accuracy(model_path, test_dataset, model_name_for_print):
    """Calculates accuracy for a model on the test set."""
    print(f"Evaluating {model_name_for_print}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3).to(device)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return 0.0

    model.eval()
    correct = 0
    total = 0
    
    # Batch processing
    batch_size = 16
    # Batch processing
    batch_size = 16
    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset[i:i+batch_size]
        inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            
        # Fix for SciBERT/BERT Label Mismatch
        # Both SciBERT and BERT-base use: 0=Neuro, 1=Mat, 2=Bio
        # We want: 0=Bio, 1=Mat, 2=Neuro (Alphabetical)
        if "scibert" in model_path.lower() or "bert-base" in model_path.lower():
             # Permute: [0, 1, 2] -> [2, 1, 0]
             probs = torch.softmax(logits, dim=1)
             probs = torch.index_select(probs, 1, torch.tensor([2, 1, 0]).to(device))
             preds = torch.argmax(probs, dim=1).cpu().numpy()
        else:
             preds = torch.argmax(logits, dim=1).cpu().numpy()
             
        labels = batch["label"]
        correct += (preds == labels).sum()
        total += len(labels)
        
    acc = correct / total
    print(f"Accuracy for {model_name_for_print}: {acc:.4f}")
    return acc

def plot_context_gain():
    print("Generating Performance Comparison...")
    
    # Load Data for Eval
    df = pd.read_csv(Config.DATA_PATH)
    df["label"] = LabelEncoder().fit_transform(df["label"])
    test_dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=Config.SEED)["test"]
    
    # 1. SciBERT
    scibert_path = "models/phase4_shootout/scibert_scivocab_uncased"
    if not os.path.exists(scibert_path): scibert_path = "allenai/scibert_scivocab_uncased"

    # 1.5 BERT-base
    bert_path = "models/phase4_shootout/bert-base-uncased"
    if not os.path.exists(bert_path): bert_path = "google-bert/bert-base-uncased"
    
    # 2. DeBERTa No Finetune
    deberta_base_path = "microsoft/deberta-v3-base"
    
    # 3. DeBERTa Finetune
    deberta_ft_path = BASELINE_MODEL_DIR
    
    # 4. DeBERTa Context
    deberta_ctx_path = "models/deberta_context_aware"
    
    # Collect Accuracies
    accuracies = {}
    
    # SciBERT
    leaderboard_path = "results/phase4_leaderboard.json"
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path) as f: p4 = json.load(f)
        # Find scibert
        sci_res = next((x for x in p4 if "scibert" in x['model']), None)
        if sci_res: accuracies["SciBERT"] = sci_res['accuracy']
        # Find BERT
        bert_res = next((x for x in p4 if "bert-base" in x['model']), None)
        if bert_res: accuracies["BERT"] = bert_res['accuracy']
    
    if "SciBERT" not in accuracies:
        accuracies["SciBERT"] = get_accuracy(scibert_path, test_dataset, "SciBERT")

    if "BERT" not in accuracies:
        accuracies["BERT"] = get_accuracy(bert_path, test_dataset, "BERT")
        
    # DeBERTa Context
    ctx_res_path = "results/phase5_context_results.json"
    if os.path.exists(ctx_res_path):
        with open(ctx_res_path) as f: res = json.load(f)
        accuracies["DeBERTa + Context"] = res['eval_accuracy']
    else:
        if os.path.exists(deberta_ctx_path):
            accuracies["DeBERTa + Context"] = get_accuracy(deberta_ctx_path, test_dataset, "DeBERTa + Context")
        else:
            accuracies["DeBERTa + Context"] = 0.0
            
    # DeBERTa Finetune
    if os.path.exists(deberta_ft_path):
        accuracies["DeBERTa (Fine-tune)"] = get_accuracy(deberta_ft_path, test_dataset, "DeBERTa (Fine-tune)")
    else:
        accuracies["DeBERTa (Fine-tune)"] = 0.0
        
    # DeBERTa No Finetune
    accuracies["DeBERTa (No FT)"] = get_accuracy(deberta_base_path, test_dataset, "DeBERTa (No FT)")

    # Plot
    data = [{"Model": k, "Accuracy": v} for k, v in accuracies.items()]
    df_res = pd.DataFrame(data)
    
    # Sort order
    order = ["DeBERTa (No FT)", "DeBERTa (Fine-tune)", "DeBERTa + Context", "SciBERT", "BERT"]
    df_res["Model"] = pd.Categorical(df_res["Model"], categories=order, ordered=True)
    df_res = df_res.sort_values("Model")
    
    plt.figure(figsize=(12,6))
    ax = sns.barplot(data=df_res, x="Model", y="Accuracy", palette="magma")
    plt.title("Phase 5: Model Performance Comparison (with BERT)", fontsize=15, fontweight='bold')
    plt.ylim(0, 1.05)
    for i in ax.containers: ax.bar_label(i, fmt='%.4f', padding=3, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase5_performance_comparison.png")
    print("Saved performance comparison.")

def plot_cluster_refinement():
    print("Generating Cluster Refinement...")
    
    df = pd.read_csv(Config.DATA_PATH)
    # Sample for visualization
    df = df.groupby('label').apply(lambda x: x.sample(n=80, random_state=42)).reset_index(drop=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define Models
    models = [
        {"name": "DeBERTa (No FT)", "path": "microsoft/deberta-v3-base"},
        {"name": "DeBERTa (Fine-tune)", "path": BASELINE_MODEL_DIR},
        {"name": "DeBERTa + Context", "path": "models/deberta_context_aware"},
        {"name": "SciBERT", "path": "models/phase4_shootout/scibert_scivocab_uncased"},
        {"name": "BERT", "path": "models/phase4_shootout/bert-base-uncased"}
    ]
    
    # Fix paths if missing
    if not os.path.exists(models[3]["path"]): models[3]["path"] = "allenai/scibert_scivocab_uncased"
    if not os.path.exists(models[4]["path"]): models[4]["path"] = "google-bert/bert-base-uncased"
    
    plt.figure(figsize=(25, 5))
    
    for i, m_info in enumerate(models):
        print(f"Embedding with {m_info['name']}...")
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
        
        plt.subplot(1, 5, i+1)
        sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=df["label"], palette="viridis", s=60, legend=False)
        plt.title(f"{m_info['name']}", fontweight='bold')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase5_cluster_refinement.png")
    print("Saved cluster refinement.")

if __name__ == "__main__":
    ensure_baseline_model()
    # plot_knowledge_map() # Removed by user request
    plot_context_gain()
    plot_cluster_refinement()
