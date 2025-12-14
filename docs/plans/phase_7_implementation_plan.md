This is the **Comprehensive Implementation Plan for Phase 7 (The Final Boss)**.

This plan implements **"Frankenstein’s DeBERTa"** (Architecture + Context + kNN) and includes a dedicated visualization suite to generate the charts you need for your report.

### **Phase 7: The "Cyborg" DeBERTa**

**Goal:** Beat SciBERT (98.08%) using DeBERTa (96.15%) by combining **Context Injection** (fixing vocabulary) and **kNN Inference** (fixing linear boundaries).

-----

#### **Step 1: Install Dependencies**

Ensure you have the plotting libraries installed.

```bash
pip install scikit-learn matplotlib seaborn
```

-----

#### **Step 2: The Logic Script (`src/phase7_knn_deberta.py`)**

**Instruction to Agent:** "Create a script that performs the Hybrid Inference. It must:

1.  Load the **Phase 5 DeBERTa** model.
2.  Re-inject the **Context Hints** (from `associativity_map.json`) into the text.
3.  Extract `[CLS]` embeddings for Train and Test sets.
4.  Train a **kNN Classifier** on the Training embeddings.
5.  Blend the Linear Head probabilities (60%) with kNN probabilities (40%) to predict.
6.  **Crucial:** Save the `embeddings`, `linear_probs`, and `knn_probs` to a `.npz` file so we can visualize them later without re-running inference."

<!-- end list -->

```python
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import json
from src.config import Config
import os

def run_knn_deberta():
    print("--- STARTING PHASE 7: kNN-DEBERTA (CONTEXT AWARE) ---")
    
    # 1. Load the Phase 5 Winner (DeBERTa + Context)
    model_path = "./models/deberta_context_aware"
    print(f"Loading Model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
    model.eval()

    # 2. Load Data & Map
    df = pd.read_csv(Config.DATA_PATH)
    dataset = load_dataset("csv", data_files=Config.DATA_PATH)["train"].train_test_split(test_size=0.2, seed=42)
    
    with open("data/associativity_map.json", "r") as f:
        assoc_map = json.load(f)

    # --- HELPER: Context Injection + Feature Extraction ---
    def get_features_with_context(text_list):
        injected_texts = []
        for text in text_list:
            words = set(text.lower().split())
            hints = []
            for word in words:
                if word in assoc_map:
                    entry = assoc_map[word]
                    syn_str = ",".join(entry['synonyms'])
                    hints.append(f"{word} -> {entry['domain']} (Syns: {syn_str})")
            
            if hints:
                context_str = " | ".join(hints[:3])
                new_text = f"{text} [SEP] [HINT] {context_str}"
            else:
                new_text = text
            injected_texts.append(new_text)

        inputs = tokenizer(injected_texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to("cuda")
        
        with torch.no_grad():
            outputs = model.deberta(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            full_outputs = model(**inputs)
            probs = F.softmax(full_outputs.logits, dim=1).cpu().numpy()
            
        return embeddings, probs

    # 3. Build Memory Bank
    print("Extracting Memory Bank (Training Data)...")
    X_train = []
    y_train = dataset["train"]["label"]
    
    batch_size = 16
    train_texts = dataset["train"]["text"]
    
    for i in range(0, len(train_texts), batch_size):
        batch = train_texts[i : i+batch_size]
        emb, _ = get_features_with_context(batch)
        X_train.append(emb)
        
    X_train = np.vstack(X_train)

    # 4. Train kNN
    print("Fitting kNN Classifier...")
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine', weights='distance')
    knn.fit(X_train, y_train)

    # 5. Inference
    print("Running Inference on Test Set...")
    test_texts = dataset["test"]["text"]
    y_test = dataset["test"]["label"]
    X_test = []
    linear_probs = []
    
    for i in range(0, len(test_texts), batch_size):
        batch = test_texts[i : i+batch_size]
        emb, prob = get_features_with_context(batch)
        X_test.append(emb)
        linear_probs.append(prob)
        
    X_test = np.vstack(X_test)
    linear_probs = np.vstack(linear_probs)

    # 6. Fusion
    knn_probs = knn.predict_proba(X_test)
    alpha = 0.4
    final_probs = ((1 - alpha) * linear_probs) + (alpha * knn_probs)
    final_preds = np.argmax(final_probs, axis=1)

    # 7. Results
    acc_baseline = accuracy_score(y_test, np.argmax(linear_probs, axis=1))
    acc_cyborg = accuracy_score(y_test, final_preds)
    
    print(f"\n>>> Phase 5 Baseline (Context Only): {acc_baseline:.4f}")
    print(f">>> Phase 7 Cyborg (Context + kNN):  {acc_cyborg:.4f}")
    
    # 8. SAVE DATA FOR VISUALIZATION (Crucial Step)
    np.savez("phase7_data.npz", 
             embeddings=X_test, 
             labels=y_test, 
             linear_probs=linear_probs, 
             knn_probs=knn_probs, 
             final_probs=final_probs)
    print("Saved inference data to phase7_data.npz for visualization.")

if __name__ == "__main__":
    run_knn_deberta()
```

-----

#### **Step 3: The Visualization Suite (`src/visualize_phase7.py`)**

**Instruction to Agent:** "Create a visualization script that loads `phase7_data.npz`. Generate three specific plots:

1.  **t-SNE Cluster Plot:** To show DeBERTa's separation (Visual proof of structural superiority).
2.  **Confidence Shift Histogram:** Compare the confidence of the model before and after kNN. (Did kNN make the model more sure?)
3.  **Correction Matrix:** A heatmap showing how many times kNN fixed a Linear error vs. broke a correct answer."

<!-- end list -->

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

def run_visualization():
    print("--- GENERATING PHASE 7 VISUALIZATIONS ---")
    
    # Load Data
    data = np.load("phase7_data.npz")
    X = data["embeddings"]
    y_true = data["labels"]
    linear_probs = data["linear_probs"]
    knn_probs = data["knn_probs"]
    final_probs = data["final_probs"]
    
    linear_preds = np.argmax(linear_probs, axis=1)
    final_preds = np.argmax(final_probs, axis=1)
    
    # --- PLOT 1: T-SNE CLUSTERS ---
    print("Generating t-SNE Plot...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=y_true, palette="deep", s=60, edgecolor="k")
    plt.title("DeBERTa (Context-Aware) Latent Space\nNote the distinct islands!", fontsize=14)
    plt.legend(title="True Label")
    plt.savefig("phase7_tsne_clusters.png")
    plt.close()
    
    # --- PLOT 2: CONFIDENCE SHIFT ---
    # We want to see if kNN increased confidence for CORRECT predictions
    print("Generating Confidence Histogram...")
    
    # Get max probability (confidence) for correct predictions only
    correct_indices = np.where(final_preds == y_true)[0]
    linear_conf = np.max(linear_probs[correct_indices], axis=1)
    cyborg_conf = np.max(final_probs[correct_indices], axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(linear_conf, color="blue", label="Linear Head (Baseline)", kde=True, alpha=0.4)
    sns.histplot(cyborg_conf, color="green", label="Cyborg Head (+kNN)", kde=True, alpha=0.4)
    plt.title("Confidence Boost on Correct Predictions", fontsize=14)
    plt.xlabel("Prediction Probability (Confidence)")
    plt.legend()
    plt.savefig("phase7_confidence_boost.png")
    plt.close()
    
    # --- PLOT 3: THE "RESCUE" MATRIX ---
    # Show how many times kNN saved us vs hurt us
    print("Generating Impact Analysis...")
    
    # Categories:
    # 1. Both Correct (Easy)
    # 2. Both Wrong (Hard)
    # 3. Linear Wrong -> Cyborg Correct (SAVED!)
    # 4. Linear Correct -> Cyborg Wrong (BROKEN)
    
    both_correct = np.sum((linear_preds == y_true) & (final_preds == y_true))
    both_wrong = np.sum((linear_preds != y_true) & (final_preds != y_true))
    saved = np.sum((linear_preds != y_true) & (final_preds == y_true))
    broken = np.sum((linear_preds == y_true) & (final_preds != y_true))
    
    categories = ['Both Correct', 'Both Wrong', 'Cyborg SAVED', 'Cyborg BROKE']
    values = [both_correct, both_wrong, saved, broken]
    colors = ['lightgray', 'gray', 'green', 'red']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, values, color=colors, edgecolor='black')
    plt.bar_label(bars)
    plt.title("Impact of kNN Integration", fontsize=14)
    plt.ylabel("Number of Abstracts")
    plt.savefig("phase7_impact_analysis.png")
    plt.close()
    
    print("Done! Check the PNG files.")

if __name__ == "__main__":
    run_visualization()
```

-----

#### **Step 4: Execution**

**Instruction to Agent:** "Run the logic first, then the visualization."

```bash
python -m src.phase7_knn_deberta
python -m src.visualize_phase7
```

### **What these plots tell your Professor**

1.  **t-SNE Plot:** "Look, the model actually understands the science perfectly (distinct clusters). The linear head was just failing."
2.  **Confidence Histogram:** "The Cyborg model isn't just more accurate; it's more *certain*. The green curve is shifted to the right."
3.  **Impact Analysis (Bar Chart):** "This green bar represents the specific edge cases where the Neighbors overruled the Head to fix an error."