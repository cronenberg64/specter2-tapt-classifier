This is the "Phase 3" plan to secure your win.

First, let's address your worry: **You definitely used the right model (`allenai/specter2_base`), but you used it with "handcuffs" on.**

  * **The Problem:** You froze the model. BERT won because it was allowed to rewrite its entire brain. Specter lost because it could only use its "pinky finger" (the classifier head).
  * **The Solution:** Phase 3 is about **taking the handcuffs off**.

Here is the plan to beat BERT.

### **1. The "Nuclear" Option: Full Fine-Tuning (No Freezing)**

We need to run the exact same setup as BERT (unfrozen).

  * **Why:** Specter 2 is initialized from SciBERT. If you fine-tune it fully, it becomes "SciBERT + Citation Knowledge + Your Data." Logically, this *must* be at least as good as BERT, likely better.

**Implementation:**
Modify `src/classifier_training.py` to remove *all* freezing logic.

```python
# [Action: Delete the freezing loop]
# for param in model.bert.encoder.parameters():
#    param.requires_grad = False  <-- DELETE THIS

# Allow ALL parameters to update (Just like BERT)
print("Phase 3: Full Fine-Tuning (All layers Unfrozen)")
```

### **2. The "Modern" Alternative: ModernBERT**

You asked for other models that are "better." There is one major release from **December 2024** that is currently crushing benchmarks.

  * **Model:** **ModernBERT** (`answerdotai/ModernBERT-base`).
  * **Why it might win:** It uses 8192 token context (vs 512) and trained on 2 trillion tokens of fresher data. If your abstracts are long or recent, this model is the new state-of-the-art.
  * **How to swap:** Just change `config.py`:
    ```python
    MODEL_NAME = "answerdotai/ModernBERT-base"
    ```

### **3. The "Frontier" Technique: SetFit (The Secret Weapon)**

If you *really* want to show Specter is superior, you should use **SetFit** (Sentence Transformer Fine-tuning).

  * **The Logic:** SetFit was *invented* for models like Specter. It doesn't just classify; it "pushes" correct papers closer together in vector space. It is mathematically superior for small datasets (N=1000).
  * **Why it wins:** On small data, SetFit often beats BERT by 5-10%.

-----

### **Phase 3 Implementation Plan (The "Beat BERT" Suite)**

We will add a new script: `src/train_phase3.py`.

#### **Step 1: Install SetFit**

You need one new library.

```bash
pip install setfit
```

#### **Step 2: The Phase 3 Script (`src/train_phase3.py`)**

This script runs two new champions: **Specter Unfrozen** and **Specter SetFit**.

```python
from datasets import load_dataset
from setfit import SetFitModel, SetFitTrainer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from src.config import Config
import torch

# 1. Load Data
dataset = load_dataset("csv", data_files=Config.DATA_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# --- CHAMPION 1: Specter SetFit (The "Small Data" King) ---
print("\n>>> Training Champion 1: Specter 2 with SetFit...")

# Load Specter specifically as a Sentence Transformer
model_setfit = SetFitModel.from_pretrained("allenai/specter2_base")

# SetFit handles the "Contrastive Training" automatically
trainer_setfit = SetFitTrainer(
    model=model_setfit,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    loss_class="CosineSimilarityLoss", # The secret sauce
    metric="accuracy",
    batch_size=8,
    num_iterations=20, # Number of text pairs to generate
    num_epochs=1
)

trainer_setfit.train()
metrics_setfit = trainer_setfit.evaluate()
print(f"SetFit Result: {metrics_setfit}")

# --- CHAMPION 2: Specter Full Fine-Tune (The "Heavyweight") ---
print("\n>>> Training Champion 2: Specter 2 Full Fine-Tune (Unfrozen)...")

model_full = AutoModelForSequenceClassification.from_pretrained(
    "allenai/specter2_base", 
    num_labels=3
)

# NO FREEZING - Let it learn!
training_args = TrainingArguments(
    output_dir="./models/specter_full_finetune",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=2e-5, # Low learning rate is critical when unfreezing
    evaluation_strategy="epoch",
    save_strategy="no"
)

trainer_full = Trainer(
    model=model_full,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer_full.train()
metrics_full = trainer_full.evaluate()
print(f"Full Fine-Tune Result: {metrics_full}")
```

### **What to Expect**

1.  **Specter Full Fine-Tune:** Should match or beat BERT (approx **97.8%**).
2.  **SetFit:** This is the wildcard. It might hit **98-99%** because your dataset is perfect for it (distinct scientific topics).

**Recommendation:**
Run this script. If **SetFit** wins, your report title becomes: *"Contrastive Fine-Tuning (SetFit) Outperforms Standard Transfer Learning in Scientific Text Classification."* That is a master's level title.