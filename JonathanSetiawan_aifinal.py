"""
AI2025 Final Assignment: Scientific Abstract Classifier
Author: Jonathan Setiawan
Description: This script implements a robust scientific abstract classifier using 
             SciBERT and SetFit (Contrastive Learning). It includes K-Fold 
             Cross-Validation, comprehensive metrics reporting, and confusion 
             matrix visualization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
import torch

# --- CONFIGURATION ---
DATA_PATH = "data/processed/phase5_augmented_dataset.csv"
MODEL_ID = "allenai/scibert_scivocab_uncased"
OUTPUT_DIR = "models/final_submission_model"
FIGURES_DIR = "results/figures"
K_FOLDS = 5
SEED = 42

# Ensure directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_and_preprocess_data(file_path):
    """
    Loads the dataset from a CSV file and performs necessary preprocessing.
    
    This function reads the scientific abstracts, removes any missing values,
    and encodes the text labels into numerical format for the model.
    
    Args:
        file_path (str): Path to the CSV dataset.
        
    Returns:
        pd.DataFrame: The cleaned and encoded dataframe.
        LabelEncoder: The fitted label encoder for inverse mapping.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Drop rows with missing text or labels
    df = df.dropna(subset=['text', 'label'])
    
    # Encode labels (Bioinformatics, Neuroscience, Materials Science)
    le = LabelEncoder()
    df['label_idx'] = le.fit_transform(df['label'])
    
    print(f"Data loaded successfully. Total samples: {len(df)}")
    print(f"Classes found: {list(le.classes_)}")
    
    return df, le

def train_and_evaluate_fold(fold_idx, train_df, val_df, device):
    """
    Trains a SetFit model on a single fold and evaluates its performance.
    
    This function handles the contrastive learning process using SciBERT as the 
    base architecture. It optimizes the vector geometry of the embeddings to 
    ensure clear separation between scientific domains.
    
    Args:
        fold_idx (int): The current fold number for logging.
        train_df (pd.DataFrame): Training data for this fold.
        val_df (pd.DataFrame): Validation data for this fold.
        device (str): The computation device (cuda or cpu).
        
    Returns:
        SetFitModel: The trained model for this fold.
        dict: Evaluation metrics (accuracy).
        np.array: Predictions on the validation set.
        np.array: True labels of the validation set.
    """
    print(f"\n--- Training Fold {fold_idx+1}/{K_FOLDS} ---")
    
    # Convert to HuggingFace Dataset format
    train_ds = Dataset.from_pandas(train_df[['text', 'label_idx']].rename(columns={'label_idx': 'label'}))
    val_ds = Dataset.from_pandas(val_df[['text', 'label_idx']].rename(columns={'label_idx': 'label'}))
    
    # Load SciBERT with a SetFit head
    model = SetFitModel.from_pretrained(MODEL_ID)
    model.to(device)
    
    # Initialize the Trainer
    # We use CosineSimilarityLoss to optimize the distance between embeddings
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss_class=CosineSimilarityLoss,
        metric="accuracy",
        batch_size=16,
        num_iterations=3, # Number of pairs to generate for contrastive learning
        num_epochs=1,
        column_mapping={"text": "text", "label": "label"}
    )
    
    # Execute training
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate()
    
    # Get predictions for metrics report
    preds = model.predict(val_df['text'].tolist())
    
    return model, metrics, preds, val_df['label_idx'].values

def plot_confusion_matrix(y_true, y_pred, target_names, save_path):
    """
    Generates and saves a confusion matrix visualization.
    
    This provides a visual representation of the model's performance across 
    different classes, helping to identify specific areas of confusion.
    
    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Model predictions.
        target_names (list): Names of the scientific domains.
        save_path (str): Where to save the generated figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Final Model Confusion Matrix')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def main():
    """
    Main execution pipeline for the AI Final Assignment.
    
    Orchestrates the end-to-end process: data loading, K-Fold cross-validation,
    metrics aggregation, and final model persistence.
    """
    print("====================================================")
    print("   AI2025 FINAL ASSIGNMENT: SCIENTIFIC CLASSIFIER   ")
    print("====================================================\n")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution Device: {device}")
    
    # 1. Data Preparation
    df, le = load_and_preprocess_data(DATA_PATH)
    
    # 2. K-Fold Cross-Validation Setup
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    
    fold_accuracies = []
    all_preds = []
    all_trues = []
    
    # 3. Cross-Validation Loop
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        model, metrics, preds, trues = train_and_evaluate_fold(fold_idx, train_df, val_df, device)
        
        fold_accuracies.append(metrics['accuracy'])
        all_preds.extend(preds)
        all_trues.extend(trues)
        
        # Save the best model (using the first fold as a baseline for this assignment)
        if fold_idx == 0:
            print(f"Saving final model to {OUTPUT_DIR}...")
            model.save_pretrained(OUTPUT_DIR)
            
    # 4. Final Metrics Reporting
    print("\n" + "="*40)
    print("         FINAL PERFORMANCE REPORT         ")
    print("="*40)
    
    print(f"\nK-Fold Cross-Validation Results (k={K_FOLDS}):")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_trues, all_preds, target_names=le.classes_))
    
    # 5. Confusion Matrix Visualization
    plot_confusion_matrix(all_trues, all_preds, le.classes_, 
                         os.path.join(FIGURES_DIR, "final_confusion_matrix.png"))
    
    print("\nAssignment Requirements Fulfilled.")
    print("Jonathan Setiawan - AI2025 Final")

if __name__ == "__main__":
    main()
