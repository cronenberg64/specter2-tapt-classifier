import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from src.config import Config
import os

def run_setfit_champion():
    print("--- STARTING PHASE 8: CONTRASTIVE LEARNING (SETFIT) ---")
    
    # 1. Load Data
    # TRAIN on Phase 5 Augmented Data (Original + Injected)
    aug_path = "data/processed/phase5_augmented_dataset.csv"
    if not os.path.exists(aug_path):
        print("Error: Augmented data not found. Run Phase 5 first.")
        return
        
    df_train = pd.read_csv(aug_path)
    # Clean NaNs
    df_train = df_train.dropna(subset=['text', 'label'])
    train_ds = Dataset.from_pandas(df_train.sample(n=min(50, len(df_train)), random_state=42))
    
    # TEST on Original Real Data (Strict Benchmark)
    # Must use same split seed=42
    df_orig = pd.read_csv(Config.DATA_PATH)
    split = Dataset.from_pandas(df_orig).train_test_split(test_size=0.2, seed=42)
    test_ds = split["test"]
    
    # Check Device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # 2. Load Architecture (DeBERTa-v3 with Safetensors)
    # Using MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli because it supports safetensors
    # and the environment blocks pickle loading.
    model_path = "./models/MoritzLaurer_DeBERTa-v3-base-mnli-fever-anli_safetensors"
    print(f"Loading DeBERTa-v3 from {model_path}...")
    
    model = SetFitModel.from_pretrained(model_path, use_safetensors=True)
    model.to(device) # Force move to device

    # 3. Setup Trainer
    # metric="accuracy" is just for logging; the real magic is the loss function
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class=CosineSimilarityLoss,
        metric="accuracy",
        batch_size=4,      # Reduced to 4 to fix CUDA OOM (6GB VRAM)
        num_iterations=1,  # Minimal iterations (1 pair per sentence) for feasibility
        num_epochs=1,      
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
    try:
        run_setfit_champion()
    except Exception as e:
        import traceback
        traceback.print_exc()
