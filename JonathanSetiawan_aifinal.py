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
from sklearn.manifold import TSNE
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

def plot_class_performance(report_dict, save_path):
    """
    Generates a bar chart comparing Precision, Recall, and F1-score for each class.
    
    Args:
        report_dict (dict): Classification report dictionary.
        save_path (str): Where to save the generated figure.
    """
    classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics = ['precision', 'recall', 'f1-score']
    
    data = []
    for cls in classes:
        for metric in metrics:
            data.append({
                'Class': cls,
                'Metric': metric.capitalize(),
                'Score': report_dict[cls][metric]
            })
            
    df_plot = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='Score', hue='Metric', data=df_plot, palette='viridis')
    plt.title('Model Performance by Class')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    print(f"Class performance plot saved to {save_path}")
    plt.close()

def plot_tsne_clusters(model_path, df, le, save_path, n_samples=1000):
    """
    Generates a t-SNE visualization of the model's embeddings.
    
    This shows how well the model separates the different scientific domains
    in its latent space.
    
    Args:
        model_path (str): Path to the saved SetFit model.
        df (pd.DataFrame): Dataframe containing text and labels.
        le (LabelEncoder): Label encoder for class names.
        save_path (str): Where to save the generated figure.
        n_samples (int): Number of samples to plot (to avoid clutter).
    """
    print("Generating t-SNE cluster visualization...")
    
    # Load model
    model = SetFitModel.from_pretrained(model_path)
    
    # Sample data
    if len(df) > n_samples:
        df_sample = df.sample(n=n_samples, random_state=42)
    else:
        df_sample = df
        
    texts = df_sample['text'].tolist()
    labels = df_sample['label_idx'].values
    class_names = le.inverse_transform(labels)
    
    # Encode
    embeddings = model.encode(texts)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=class_names, palette='deep', s=60, edgecolor='k')
    plt.title('t-SNE Visualization of Scientific Abstract Embeddings')
    plt.legend(title='Domain')
    plt.savefig(save_path)
    print(f"t-SNE plot saved to {save_path}")
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
    
    # Generate report dict for visualization
    report_dict = classification_report(all_trues, all_preds, target_names=le.classes_, output_dict=True)
    
    # 5. Visualizations
    print("\nGenerating Visualizations...")
    
    # Confusion Matrix
    plot_confusion_matrix(all_trues, all_preds, le.classes_, 
                         os.path.join(FIGURES_DIR, "final_confusion_matrix.png"))
                         
    # Class Performance
    plot_class_performance(report_dict, 
                          os.path.join(FIGURES_DIR, "final_class_performance.png"))
                          
    # t-SNE Clusters (using the saved best model)
    plot_tsne_clusters(OUTPUT_DIR, df, le, 
                      os.path.join(FIGURES_DIR, "final_tsne_clusters.png"))
    
    print("\nAssignment Requirements Fulfilled.")
    print("Jonathan Setiawan - AI2025 Final")

if __name__ == "__main__":
    main()
