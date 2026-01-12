"""
Script to generate additional visualizations for the Final Assignment.
Run this AFTER JonathanSetiawan_aifinal.py has finished training.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from setfit import SetFitModel

# --- CONFIGURATION ---
DATA_PATH = "data/processed/augmented_dataset.csv"
MODEL_PATH = "models/final_submission_model"
FIGURES_DIR = "results/figures"

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['text', 'label'])
    le = LabelEncoder()
    df['label_idx'] = le.fit_transform(df['label'])
    return df, le

def plot_class_performance(report_dict, accuracy, save_path):
    print("Generating Class Performance Plot...")
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
    ax = sns.barplot(x='Class', y='Score', hue='Metric', data=df_plot, palette='viridis')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3, rotation=90)
        
    plt.title(f'Model Performance by Class (Overall Accuracy: {accuracy:.2%})')
    plt.ylim(0, 1.15) # Increase limit to make room for labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved to {save_path}")
    plt.close()

def plot_tsne_clusters(model, df, le, save_path, n_samples=1000):
    print("Generating t-SNE Cluster Visualization...")
    
    if len(df) > n_samples:
        df_sample = df.sample(n=n_samples, random_state=42)
    else:
        df_sample = df
        
    texts = df_sample['text'].tolist()
    labels = df_sample['label_idx'].values
    class_names = le.inverse_transform(labels)
    
    print("Encoding texts...")
    embeddings = model.encode(texts)
    
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=class_names, palette='deep', s=60, edgecolor='k')
    plt.title('t-SNE Visualization of Scientific Abstract Embeddings')
    plt.legend(title='Domain')
    plt.savefig(save_path)
    print(f"Saved to {save_path}")
    plt.close()

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please wait for JonathanSetiawan_aifinal.py to finish training first.")
        return

    # 1. Load Data
    df, le = load_data(DATA_PATH)
    
    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = SetFitModel.from_pretrained(MODEL_PATH)
    
    # 3. Generate Predictions for Metrics
    print("Generating predictions...")
    # Use the model's encode method to get embeddings first (batched)
    # SetFit predict usually handles this, but for progress bar we can be explicit or just trust the internal batching
    # Let's use a simple loop with tqdm if predict doesn't support it directly in this version
    from tqdm import tqdm
    
    batch_size = 32
    texts = df['text'].tolist()
    preds = []
    
    print(f"Predicting on {len(texts)} samples in batches of {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_preds = model.predict(batch)
        preds.extend(batch_preds)
        
    true_labels = df['label_idx'].values
    
    # 4. Generate Classification Report
    print("\n" + "="*40)
    print("         MODEL PERFORMANCE REPORT         ")
    print("="*40)
    
    # Calculate Accuracy
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(true_labels, preds)
    print(f"\nOverall Accuracy: {acc:.4f} ({acc:.2%})")
    
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, preds, target_names=le.classes_))
    
    report_dict = classification_report(true_labels, preds, target_names=le.classes_, output_dict=True)
    
    # 5. Plot Class Performance
    plot_class_performance(report_dict, acc, os.path.join(FIGURES_DIR, "final_class_performance.png"))
    
    # 6. Plot t-SNE
    plot_tsne_clusters(model, df, le, os.path.join(FIGURES_DIR, "final_tsne_clusters.png"))
    
    print("\nVisualizations generated successfully!")

if __name__ == "__main__":
    main()
