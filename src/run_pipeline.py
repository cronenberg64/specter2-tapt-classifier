from src.collect_data import fetch_and_save_data
from src.tapt_training import run_tapt
from src.classifier_training import run_training
from src.evaluation import evaluate_all_models
import os

def main():
    # 1. Data
    if not os.path.exists("./data/raw/scientific_abstracts_dataset.csv"):
        print("Fetching data...")
        fetch_and_save_data()
    else:
        print("Data already exists.")
    
    # 2. TAPT (Only for SPECTER)
    # Check if TAPT is already done to save time
    if not os.path.exists("./models/specter_tapt"):
        print("Running TAPT...")
        run_tapt()
    else:
        print("TAPT model already exists.")
    
    # 3. Train Classifiers
    models_to_train = ["specter_tapt", "bert", "roberta"]
    for model in models_to_train:
        print(f"--- Training {model} ---")
        run_training(model_name=model)
    
    # 4. Evaluation
    evaluate_all_models()

if __name__ == "__main__":
    main()
