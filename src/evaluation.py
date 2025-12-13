import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import Config
import numpy as np

def evaluate_all_models():
    print("--- Starting Evaluation ---")
    
    # 1. Load Test Data (Same split logic to ensure we get the same validation set)
    # Ideally we should have saved the split, but for this demo we use random_state
    df = pd.read_csv(Config.DATA_PATH)
    
    label_list = df['label'].unique().tolist()
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    df['labels'] = df['label'].map(label2id)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.SEED)
    for train_index, val_index in sss.split(df, df['label']):
        val_df = df.iloc[val_index]
    
    val_dataset = Dataset.from_pandas(val_df)
    
    models_to_eval = ["specter_tapt", "bert", "roberta"]
    results = []
    
    os.makedirs("results", exist_ok=True)
    
    for model_name in models_to_eval:
        model_path = f"./models/{model_name}_classifier"
        if not os.path.exists(model_path):
            print(f"Skipping {model_name} (not found)")
            continue
            
        print(f"Evaluating {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        def tokenize(batch):
            return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=Config.MAX_LENGTH)
            
        # Remove 'text' and 'label' columns, keep 'label_id' (renamed to labels if needed, but here we used label_id in df)
        # Wait, in evaluation.py we have: df['label_id'] = ...
        # But Trainer expects 'labels'.
        # We should rename 'label_id' to 'labels' in df first.
        
        cols_to_remove = [c for c in val_dataset.column_names if c != 'labels']
        tokenized_val = val_dataset.map(tokenize, batched=True, remove_columns=cols_to_remove)
        
        trainer = Trainer(model=model)
        preds_output = trainer.predict(tokenized_val)
        preds = preds_output.predictions.argmax(-1)
        labels = preds_output.label_ids
        
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        
        print(f"  {model_name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "F1 Score": f1
        })
        
        # Confusion Matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_list, yticklabels=label_list)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f"results/confusion_matrix_{model_name}.png")
        plt.close()
        
    # Save Results CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/comparison_report.csv", index=False)
    
    # Combined Bar Chart
    if not results_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='F1 Score', data=results_df)
        plt.title('Model Comparison (F1 Score)')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig("results/model_comparison.png")
        plt.close()
        
    print("Evaluation Complete. Results saved to 'results/' directory.")

if __name__ == "__main__":
    evaluate_all_models()
