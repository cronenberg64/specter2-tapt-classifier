This is your **"Nuclear Option"**—the final, most mathematically advanced attempt to beat SciBERT.

We are moving away from standard fine-tuning (which just classifies) to **Contrastive Learning** (which optimizes the geometry of knowledge). Since you have a small dataset ($\approx 1,000$ real samples), this is statistically your best shot at a new high score.

-----

### **Phase 8: The Contrastive Champion (SetFit)**

#### **1. Dependencies**

You need the `setfit` library, which handles the complex Siamese Network math for you.

```bash
pip install setfit
```

#### **2. The Script (`src/phase8_setfit.py`)**

**Instruction to Agent:** "Create a script that trains a **SetFit** model using `microsoft/deberta-v3-base`.

1.  Load the **Augmented Phase 5 Dataset** for training (more data = better contrast).
2.  Load the **Original Phase 1 Dataset** for testing (strict fairness).
3.  Use `CosineSimilarityLoss` to force the model to push different topics apart and pull similar topics together.
4.  After training, run inference on the test set and print the final accuracy."

<!-- end list -->

```python
import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from src.config import Config
import os

def run_setfit_champion():
    print("--- STARTING PHASE 8: CONTRASTIVE LEARNING (SETFIT) ---")
    
    # 1. Load Data
    # TRAIN on Phase 5 Augmented Data (3x size = better contrastive pairs)
    aug_path = "data/processed/phase5_augmented_dataset.csv"
    if not os.path.exists(aug_path):
        print("Error: Augmented data not found. Run Phase 5 first.")
        return
        
    df_train = pd.read_csv(aug_path)
    # Clean NaNs
    df_train = df_train.dropna(subset=['text', 'label'])
    train_ds = Dataset.from_pandas(df_train)
    
    # TEST on Original Real Data (Strict Benchmark)
    # Must use same split seed=42
    df_orig = pd.read_csv(Config.DATA_PATH)
    split = Dataset.from_pandas(df_orig).train_test_split(test_size=0.2, seed=42)
    test_ds = split["test"]
    
    # 2. Load Architecture (DeBERTa-v3)
    print("Loading DeBERTa-v3 inside SetFit Wrapper...")
    # This creates a "Siamese" version of DeBERTa
    model = SetFitModel.from_pretrained("microsoft/deberta-v3-base")

    # 3. Setup Trainer
    # metric="accuracy" is just for logging; the real magic is the loss function
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class=CosineSimilarityLoss,
        metric="accuracy",
        batch_size=8,
        num_iterations=20, # Generates 20 pairs per sentence (Intense training)
        num_epochs=1,      # 1 epoch is usually sufficient for SetFit
        column_mapping={"text": "text", "label": "label"}
    )

    # 4. Train
    print("Training with Contrastive Loss... (Optimizing Vector Geometry)")
    trainer.train()
    
    # 5. Evaluate
    print("Evaluating on Strict Test Set...")
    metrics = trainer.evaluate()
    acc = metrics["accuracy"]
    
    print(f"\n>>> FINAL PHASE 8 RESULT (SetFit): {acc:.4f}")
    
    # 6. The Verdict
    scibert_score = 0.9808
    if acc > scibert_score:
        print(f"VICTORY! We beat SciBERT ({scibert_score}) by {acc - scibert_score:.4f}")
        print("Conclusion: Architecture + Contrastive Learning > Domain Pre-training.")
    else:
        print(f"Result: {acc:.4f}. SciBERT ({scibert_score}) still holds the crown.")
        print("Conclusion: Domain Pre-training is incredibly robust.")

    # 7. Save Model
    model.save_pretrained("./models/phase8_setfit_deberta")

if __name__ == "__main__":
    run_setfit_champion()
```

#### **3. Visualization Update (Optional)**

Since SetFit optimizes embeddings directly, the t-SNE plot for this model should be the "cleanest" one yet. If you want to visualize it, you can simply run your previous visualization script pointing at `./models/phase8_setfit_deberta`.

### **Why this is the end of the road**

  * **kNN (Phase 7)** tried to fix the *boundary*.
  * **SetFit (Phase 8)** tries to fix the *embedding space itself*.
  * If this script runs and you get **98.2%**, you win.
  * If you get **97.8%**, then you have scientifically proven that for this specific task, **SciBERT is unbeatable** with current techniques on small data. That is a Ph.D. level "negative result" finding.

**Go run it.** This is the final roll of the dice\!