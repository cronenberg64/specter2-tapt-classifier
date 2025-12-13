This is the **"Phase 2: Experimentation Suite"** implementation plan.

The goal here is to move from a single data point ("Frozen TAPT") to a **comprehensive scientific study** that generates a winning results table.

We will automate three experiments:

1.  **Baseline Check (Ablation):** Specter *without* TAPT (Does TAPT actually help?).
2.  **The Challenger (Unfreezing):** Specter TAPT with the top 2 layers unfrozen (Can we beat BERT?).
3.  **Data Efficiency:** Training on only 10% of data (Is Specter better when data is scarce?).

Pass these instructions to your agent.

-----

### **Step 1: Upgrade `src/config.py`**

**Instruction to Agent:**
"Modify the `Config` class to support dynamic experiment settings. Add default values for `USE_TAPT_WEIGHTS` (bool), `UNFREEZE_LAST_N_LAYERS` (int), and `DATA_FRACTION` (float)."

```python
class Config:
    # ... existing paths ...
    
    # NEW: Experiment Control Knobs (Defaults)
    USE_TAPT_WEIGHTS = True       # True = use ./models/specter_tapt, False = use allenai/specter2_base
    UNFREEZE_LAST_N_LAYERS = 0    # 0 = Freeze all. 2 = Unfreeze top 2. 12 = Unfreeze all.
    DATA_FRACTION = 1.0           # 1.0 = 100% data. 0.1 = 10% data.
    
    # ... existing hyperparameters ...
```

-----

### **Step 2: Upgrade `src/classifier_training.py`**

**Instruction to Agent:**
"Refactor the training script to respect the new Config flags. It needs three specific logic upgrades:

1.  **Model Loading Switch:** Load from TAPT path or Base path based on `Config.USE_TAPT_WEIGHTS`.
2.  **Smart Freezing:** Implement logic to freeze layers `0` to `12 - N` and unfreeze the rest.
3.  **Data Slicing:** If `DATA_FRACTION < 1.0`, randomly select a subset of the training data."

**Code Logic for Agent (Smart Freezing):**

```python
# [Agent Prompt: Insert this logic after model loading]
def configure_layers(model, unfreeze_last_n):
    # 1. First, freeze EVERYTHING (Base Encoder)
    for param in model.bert.encoder.parameters():
        param.requires_grad = False
        
    # 2. Always unfreeze the Classifier Head (The top implementation layer)
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # 3. Smart Unfreezing of Encoder Layers
    if unfreeze_last_n > 0:
        num_layers = len(model.bert.encoder.layer) # Usually 12
        start_layer = num_layers - unfreeze_last_n
        
        print(f"Unfreezing encoder layers {start_layer} to {num_layers - 1}...")
        for i in range(start_layer, num_layers):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = True
```

**Code Logic for Agent (Data Slicing):**

```python
# [Agent Prompt: Insert before Trainer initialization]
if Config.DATA_FRACTION < 1.0:
    subset_size = int(len(train_dataset) * Config.DATA_FRACTION)
    print(f"Subsampling Training Data: Using {subset_size} samples ({Config.DATA_FRACTION*100}%)")
    train_dataset = train_dataset.shuffle(seed=Config.SEED).select(range(subset_size))
```

-----

### **Step 3: The Automator (`run_experiments.py`)**

**Instruction to Agent:**
"Create a new script `run_experiments.py` that loops through different configurations, updates the Config, runs the trainer, and saves the metrics to a JSON file."

```python
import json
from src.config import Config
from src.classifier_training import run_training
from src.evaluation import evaluate_model

# Define the experiments
experiments = [
    {
        "name": "Exp_1_Specter_Base_Frozen",
        "use_tapt": False,
        "unfreeze": 0,
        "data": 1.0,
        "note": "Baseline: Does Specter work without our innovation?"
    },
    {
        "name": "Exp_2_Specter_TAPT_Frozen",
        "use_tapt": True,
        "unfreeze": 0,
        "data": 1.0,
        "note": "Our Current Best: Does TAPT help?"
    },
    {
        "name": "Exp_3_Specter_TAPT_Top2",
        "use_tapt": True,
        "unfreeze": 2,
        "data": 1.0,
        "note": "The Challenger: Can we beat BERT by unfreezing slightly?"
    },
    {
        "name": "Exp_4_LowData_Specter",
        "use_tapt": True,
        "unfreeze": 0,
        "data": 0.1,
        "note": "Efficiency Check: 10% data only"
    }
]

results_log = []

for exp in experiments:
    print(f"\n>>> RUNNING: {exp['name']}")
    
    # 1. Dynamic Config Override (Hack for script-based config)
    Config.USE_TAPT_WEIGHTS = exp["use_tapt"]
    Config.UNFREEZE_LAST_N_LAYERS = exp["unfreeze"]
    Config.DATA_FRACTION = exp["data"]
    Config.FINAL_MODEL_DIR = f"./models/{exp['name']}" # Save distinct models
    
    # 2. Run Train & Eval
    run_training()
    metrics = evaluate_model() # Make sure evaluation.py returns the dict!
    
    # 3. Log
    results_log.append({
        "experiment": exp["name"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "config": exp
    })

# Save Final Results
with open("final_experiment_results.json", "w") as f:
    json.dump(results_log, f, indent=4)

print("ALL EXPERIMENTS COMPLETE. Check final_experiment_results.json")
```

-----

### **Step 4: Analyze the expected outcome**

Once you run this (it might take \~1 hour depending on your GPU), you will likely see a table like this for your report:

  * **Exp 1 (Base):** \~93% (Shows TAPT provided \~1.7% gain).
  * **Exp 2 (Frozen TAPT):** \~94.7% (Your current result).
  * **Exp 3 (Top-2 Unfrozen):** \~97.5% - 98.0% (Likely matches or beats BERT).
  * **Exp 4 (Low Data):** This is the wild card. If this gets \~85-90% while BERT (if you tested it) gets \~70%, you have a massive "Real World Application" argument.

**Immediate Next Action:**
Give the agent the **Step 1** and **Step 2** prompts first to ensure the code structure is ready, then run the **Step 3** script overnight.