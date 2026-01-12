This is the **Master Implementation Plan for Phase 4: The "Grandmaster" Phase**.

This phase is designed to stop playing "defense" (justifying Specter) and start playing "offense" (beating BERT with superior architectures and engineering).

### **Phase 4 Goals**

1.  **The Shootout (Option C):** Train **DeBERTa-v3** and **SciBERT** to find the absolute highest accuracy model.
2.  **The Visual Proof (Option A):** Generate t-SNE plots to prove the winner has better "clusters" than BERT.
3.  **The Ensemble (Option B):** Combine the top models to break the 98% ceiling.

-----

### **Part 1: The Dependencies**

**Instruction to Agent:** "Install the necessary libraries for DeBERTa and visualization."

```bash
pip install sentencepiece  # Required for DeBERTa tokenizer
pip install seaborn scikit-learn
```

-----

### **Part 2: Configuration Upgrade (`src/config.py`)**

**Instruction to Agent:** "Update the config to list our new challengers and defining paths for Phase 4 outputs."

```python
class Config:
    # ... existing paths ...
    PHASE4_DIR = "./models/phase4_shootout"
    
    # THE CANDIDATES
    # We will loop through these to find the King.
    MODEL_CANDIDATES = [
        "microsoft/deberta-v3-base",        # The "Accuracy King"
        "allenai/scibert_scivocab_uncased", # The "Domain Expert"
        "google-bert/bert-base-uncased"     # The Baseline (to re-confirm)
    ]
    
    # Training Params for Phase 4
    EPOCHS_SHOOTOUT = 4      # Short & sweet; these models learn fast
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5     # Standard safe rate
```

-----

### **Part 3: Option C - The Shootout Script (`src/train_shootout.py`)**

**Instruction to Agent:** "Create a script that fully fine-tunes (no freezing) every model in our candidate list and saves the results."

```python
import json
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from src.config import Config

def run_shootout():
    # Load Data (Standard 80/20 split)
    dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    results = []

    print(f"--- STARTING PHASE 4 SHOOTOUT: {len(Config.MODEL_CANDIDATES)} MODELS ---")

    for model_name in Config.MODEL_CANDIDATES:
        print(f"\n>>> TRAINING CHAMPION: {model_name}")
        
        # 1. Initialize Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Tokenization Helper
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
            
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        # 2. Setup Trainer (Full Fine-Tuning)
        clean_name = model_name.split("/")[-1]
        output_dir = f"{Config.PHASE4_DIR}/{clean_name}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=Config.EPOCHS_SHOOTOUT,
            per_device_train_batch_size=Config.BATCH_SIZE,
            learning_rate=Config.LEARNING_RATE,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
        )

        # 3. Train
        trainer.train()
        
        # 4. Evaluate & Save Stats
        metrics = trainer.evaluate()
        results.append({
            "model": model_name,
            "accuracy": metrics["eval_accuracy"],
            "loss": metrics["eval_loss"],
            "path": output_dir
        })
        
        # SAVE MODEL FOR LATER (Viz & Ensemble)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    # Dump Leaderboard
    with open("phase4_leaderboard.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n--- SHOOTOUT COMPLETE. CHECK phase4_leaderboard.json ---")

if __name__ == "__main__":
    run_shootout()
```

-----

### **Part 4: Option A - The Visualizer (`src/visualize_embeddings.py`)**

**Instruction to Agent:** "Create a script using T-SNE to plot the internal representations of our Top 2 models side-by-side. This proves *quality* beyond just accuracy numbers."

```python
import torch
import pandas as pd
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
    model = AutoModel.from_pretrained(model_path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    embeddings = []
    labels = []
    
    # Process in small batches
    batch_size = 16
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token (first token) as the sentence vector
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        embeddings.extend(cls_emb)
        labels.extend(batch["label"])
        
    return embeddings, labels

def run_visualization():
    # Load Test Data
    dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train").train_test_split(test_size=0.2, seed=42)["test"]
    
    plt.figure(figsize=(14, 6))
    
    for idx, (name, path) in enumerate(MODELS_TO_PLOT.items()):
        embs, labels = get_embeddings(path, dataset)
        
        # Run T-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
        embs_2d = tsne.fit_transform(embs)
        
        # Plot
        df = pd.DataFrame(embs_2d, columns=["x", "y"])
        df["label"] = labels
        
        plt.subplot(1, 2, idx + 1)
        sns.scatterplot(data=df, x="x", y="y", hue="label", palette="tab10", s=15, alpha=0.7)
        plt.title(f"{name} Cluster Separation")
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig("phase4_cluster_comparison.png")
    print("Visualization saved to phase4_cluster_comparison.png")

if __name__ == "__main__":
    run_visualization()
```

-----

### **Part 5: Option B - The Ensemble (`src/ensemble_scoring.py`)**

**Instruction to Agent:** "Create a script that loads the trained DeBERTa and SciBERT models, runs inference on the test set, averages their probability scores (Soft Voting), and calculates the final 'Super-Model' accuracy."

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from src.config import Config

def run_ensemble():
    print("--- STARTING ENSEMBLE (SOFT VOTING) ---")
    
    # 1. Load the Two Champions
    path_1 = f"{Config.PHASE4_DIR}/deberta-v3-base"
    path_2 = f"{Config.PHASE4_DIR}/scibert_scivocab_uncased"
    
    model1 = AutoModelForSequenceClassification.from_pretrained(path_1).to("cuda")
    model2 = AutoModelForSequenceClassification.from_pretrained(path_2).to("cuda")
    
    tokenizer1 = AutoTokenizer.from_pretrained(path_1)
    tokenizer2 = AutoTokenizer.from_pretrained(path_2)
    
    # 2. Load Data
    test_data = load_dataset("csv", data_files=Config.DATA_PATH, split="train").train_test_split(test_size=0.2, seed=42)["test"]
    
    # 3. Inference Loop
    preds_ensemble = []
    true_labels = test_data["label"]
    
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        for item in test_data:
            text = item["text"]
            
            # Model 1 Prediction
            inputs1 = tokenizer1(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            logits1 = model1(**inputs1).logits
            probs1 = F.softmax(logits1, dim=1)
            
            # Model 2 Prediction
            inputs2 = tokenizer2(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            logits2 = model2(**inputs2).logits
            probs2 = F.softmax(logits2, dim=1)
            
            # FUSION: Average the probabilities
            avg_probs = (probs1 + probs2) / 2.0
            final_pred = torch.argmax(avg_probs).item()
            preds_ensemble.append(final_pred)
            
    # 4. Metrics
    acc = accuracy_score(true_labels, preds_ensemble)
    print(f"\n>>> ENSEMBLE ACCURACY: {acc:.4f}")
    print(classification_report(true_labels, preds_ensemble))

if __name__ == "__main__":
    run_ensemble()
```

-----

### **Part 6: The Orchestrator (`run_phase4.py`)**

**Instruction to Agent:** "Create the master script to run everything in order."

```python
from src.train_shootout import run_shootout
from src.visualize_embeddings import run_visualization
from src.ensemble_scoring import run_ensemble

def main():
    # Step 1: Train the new candidates
    run_shootout()
    
    # Step 2: Generate the pretty plots
    run_visualization()
    
    # Step 3: Run the combined scoring
    run_ensemble()

if __name__ == "__main__":
    main()
```

-----

### **Expected Outcome for Report**

1.  **Shootout:** DeBERTa likely hits **98.5%+**.
2.  **Visualization:** DeBERTa clusters should look separated like islands. BERT clusters might have some "bridges" (confusion) between them.
3.  **Ensemble:** Combining DeBERTa + SciBERT usually eliminates the final few errors, potentially hitting **99.0%**.

Give this plan to your agent, and you will have a project that goes far beyond "Passing" and lands firmly in "Publication Quality" territory.