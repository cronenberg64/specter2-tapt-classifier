import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
    # Note: run_training takes model_name arg, but here we are overriding Config.
    # The script uses model_name to decide logic.
    # We should pass "specter_tapt" to trigger the Specter logic branch in classifier_training.py
    # because our Config overrides will handle the specifics (TAPT vs Base, Freezing).
    run_training(model_name="specter_tapt")
    
    metrics = evaluate_model(model_path=Config.FINAL_MODEL_DIR)
    
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
