import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import Config
except ImportError:
    class Config:
        DATA_PATH = "data/raw/scientific_abstracts_dataset.csv"

def train_bert_base_setfit():
    print("--- TRAINING BERT-base with SetFit (Phase 8 Fair Duel) ---")
    
    output_dir = "./models/phase8_bert-base_body"
    
    # Check if already trained
    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"Model already exists at {output_dir}. Skipping training.")
        return
    
    # 1. Load Data
    aug_path = "data/processed/phase5_augmented_dataset.csv"
    if not os.path.exists(aug_path):
        print(f"Warning: Augmented data not found at {aug_path}. Using original data.")
        aug_path = Config.DATA_PATH
        
    df_train = pd.read_csv(aug_path)
    df_train = df_train.dropna(subset=['text', 'label'])
    
    # Sample for training (same as SciBERT/BERT-Large)
    train_ds = Dataset.from_pandas(df_train.sample(n=min(50, len(df_train)), random_state=42))
    
    # Test on Original Real Data
    df_orig = pd.read_csv(Config.DATA_PATH)
    split = Dataset.from_pandas(df_orig).train_test_split(test_size=0.2, seed=42)
    test_ds = split["test"]
    
    # Check Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # 2. Load BERT-base from FINE-TUNED checkpoint (same treatment as SciBERT/BERT-Large)
    finetuned_path = "models/phase4_shootout/bert-base-uncased"
    print(f"Loading FINE-TUNED BERT-base from {finetuned_path} for SetFit...")
    model = SetFitModel.from_pretrained(finetuned_path)
    model.to(device)

    # 3. Setup Trainer (same config as SciBERT/BERT-Large)
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class=CosineSimilarityLoss,
        metric="accuracy",
        batch_size=4,
        num_iterations=1,
        num_epochs=1,
        column_mapping={"text": "text", "label": "label"}
    )

    # 4. Train
    print("Training with Contrastive Loss...")
    trainer.train()
    
    # 5. Evaluate
    print("Evaluating on Strict Test Set...")
    metrics = trainer.evaluate()
    acc = metrics["accuracy"]
    
    print(f"\n>>> BERT-base SetFit Result: {acc:.4f}")
    
    # 6. Save Model
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    try:
        train_bert_base_setfit()
    except Exception as e:
        import traceback
        traceback.print_exc()
