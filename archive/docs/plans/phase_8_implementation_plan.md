Here is the updated, finalized implementation plan for **Phase 8**.

This plan executes the **"Fair Duel"**: You will train **both** the Challenger (DeBERTa) and the Champion (SciBERT) using the **SetFit Contrastive Learning** pipeline. This ensures that if DeBERTa wins, it wins because of its architecture, not just because of the method.

Given your powerful hardware (RTX A4000), I have optimized the batch sizes for speed.

### **Phase 8: The "Fair Duel" (Contrastive Learning)**

**Goal:** Determine if Generalist Architecture (DeBERTa) can beat Domain Pre-training (SciBERT) when both are optimized using SOTA Contrastive Learning on augmented data.

-----

#### **Step 1: Install Dependencies**

```bash
pip install setfit matplotlib seaborn scikit-learn
```

-----

#### **Step 2: The Fair Duel Script (`src/phase8_fair_duel.py`)**

**What this does:**

1.  Loads the **Augmented Data** (Phase 5) for training (3x data volume).
2.  Loads the **Original Data** for testing (Strict Benchmark).
3.  Trains **SciBERT** using SetFit (Baseline for this phase).
4.  Trains **DeBERTa** using SetFit (Challenger).
5.  Optimizes using `CosineSimilarityLoss` (Geometry optimization).
6.  Saves both models for visualization.

<!-- end list -->

```python
import pandas as pd
import os
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from src.config import Config

def run_fair_duel():
    print("--- STARTING PHASE 8: THE FAIR DUEL (SetFit vs SetFit) ---")
    
    # 1. Load Data
    # TRAIN: Phase 5 Augmented Data (Rich signal)
    aug_path = "data/processed/phase5_augmented_dataset.csv"
    if not os.path.exists(aug_path):
        print("Error: Phase 5 data not found. Run 'python -m src.phase5_augment' first.")
        return

    df_train = pd.read_csv(aug_path).dropna(subset=['text', 'label'])
    train_ds = Dataset.from_pandas(df_train)
    
    # TEST: Original Phase 1 Data (Strict Benchmark)
    df_orig = pd.read_csv(Config.DATA_PATH)
    test_ds = Dataset.from_pandas(df_orig).train_test_split(test_size=0.2, seed=42)["test"]
    
    print(f"Training on {len(train_ds)} samples. Testing on {len(test_ds)} samples.")

    # 2. Define Contenders
    contenders = {
        "SciBERT": "allenai/scibert_scivocab_uncased",
        "DeBERTa": "microsoft/deberta-v3-base"
    }
    
    results = {}

    for name, model_id in contenders.items():
        print(f"\n>>> ENTERING ARENA: {name} ({model_id})")
        
        # Load Model (Siamese Head)
        model = SetFitModel.from_pretrained(model_id)
        
        # Setup Trainer
        # Hardware Opt: Batch size 32 is safe for A4000 (16GB VRAM)
        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            loss_class=CosineSimilarityLoss,
            metric="accuracy",
            batch_size=32,     # Optimized for your GPU
            num_iterations=20, # Generates 20 pairs per sample (60k+ total steps)
            num_epochs=1,      # 1 epoch is standard for SetFit
            column_mapping={"text": "text", "label": "label"}
        )
        
        # Train
        print(f"Training {name} with Contrastive Loss...")
        trainer.train()
        
        # Evaluate
        metrics = trainer.evaluate()
        score = metrics["accuracy"]
        results[name] = score
        print(f"Result for {name}: {score:.4f}")
        
        # Save Model for Visualization
        save_path = f"./models/phase8_{name.lower()}"
        model.save_pretrained(save_path)
        print(f"Saved {name} to {save_path}")

    # 3. Final Scoreboard
    print("\n\n=== FINAL PHASE 8 SCOREBOARD ===")
    print(f"Previous SOTA (Standard SciBERT): 0.9808")
    print("-" * 40)
    for name, score in results.items():
        print(f"{name} (SetFit): {score:.4f}")
    
    # The Verdict
    if results["DeBERTa"] > 0.9808:
        print("\nVICTORY! DeBERTa + SetFit has beaten the Domain Specialist.")
    else:
        print("\nRESULT: Domain Knowledge prevails, but check if SciBERT (SetFit) set a new record.")

if __name__ == "__main__":
    run_fair_duel()
```

-----

#### **Step 3: The Visualization Script (`src/visualize_phase8.py`)**

Since SetFit optimizes embeddings specifically for clustering, your **t-SNE plots** should look even cleaner than Phase 4. This script compares the embedding space of both models side-by-side.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from setfit import SetFitModel
from datasets import load_dataset
from src.config import Config

def visualize_fair_duel():
    print("--- GENERATING PHASE 8 VISUALIZATIONS ---")
    
    # Load Test Data
    df = pd.read_csv(Config.DATA_PATH)
    dataset = load_dataset("csv", data_files=Config.DATA_PATH)["train"].train_test_split(test_size=0.2, seed=42)["test"]
    texts = dataset["text"]
    labels = dataset["label"]
    
    models_to_plot = ["SciBERT", "DeBERTa"]
    
    plt.figure(figsize=(20, 8))
    
    for i, name in enumerate(models_to_plot):
        path = f"./models/phase8_{name.lower()}"
        print(f"Loading {name} from {path}...")
        
        try:
            model = SetFitModel.from_pretrained(path)
            # Encode (Get Embeddings)
            embeddings = model.encode(texts)
            
            # t-SNE
            print(f"Running t-SNE for {name}...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto")
            X_2d = tsne.fit_transform(embeddings)
            
            # Plot
            plt.subplot(1, 2, i+1)
            sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=labels, palette="deep", s=60, edgecolor="k")
            plt.title(f"{name} (SetFit) Latent Space", fontsize=16)
            plt.legend(title="Class")
            
        except Exception as e:
            print(f"Skipping {name}: {e}")

    plt.tight_layout()
    plt.savefig("phase8_comparison.png")
    print("Saved comparison plot to phase8_comparison.png")

if __name__ == "__main__":
    visualize_fair_duel()
```

-----

#### **Step 4: Execution**

1.  **Run the Training:**

    ```bash
    python -m src.phase8_fair_duel
    ```

    *(Estimated time on your RTX A4000: \~20-30 minutes total)*

2.  **Run the Visualization:**

    ```bash
    python -m src.visualize_phase8
    ```

### **What to look for**

  * **If SciBERT jumps to \~99%:** You write: "Contrastive Learning turns a domain specialist into a near-perfect classifier."
  * **If DeBERTa hits \~98.5%:** You write: "Generalist Architecture + Contrastive Learning effectively replicates Domain Pre-training."

This is the scientifically rigorous end to your project. Good luck\!