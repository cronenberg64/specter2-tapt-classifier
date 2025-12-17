import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
import os
from src.config import Config

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1,
    }

def run_training(model_name):
    print(f"--- Starting training for {model_name} ---")
    
    # 1. Load Data
    df = pd.read_csv(Config.DATA_PATH)
    
    # Encode labels
    label_list = df['label'].unique().tolist()
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    df['labels'] = df['label'].map(label2id)
    
    # 2. Split Data (Stratified)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.SEED)
    for train_index, val_index in sss.split(df, df['labels']):
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # 3. Model & Tokenizer Selection
    if model_name == "specter_tapt":
        if Config.USE_TAPT_WEIGHTS:
            model_path = Config.TAPT_OUTPUT_DIR
            print("Using TAPT weights...")
        else:
            model_path = Config.MODEL_NAME
            print("Using Base Specter 2 weights (No TAPT)...")
    elif model_name == "bert":
        model_path = "bert-base-uncased"
    elif model_name == "roberta":
        model_path = "roberta-base"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
        
    print(f"Loading tokenizer from {model_path}...")
    print(f"Absolute path: {os.path.abspath(model_path)}")
    if os.path.exists(model_path):
        print(f"Contents: {os.listdir(model_path)}")
    else:
        print(f"Directory not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True
    )

    # 4. Tokenization
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=Config.MAX_LENGTH)
        
    # Remove 'text' and 'label' columns, keep 'labels'
    cols_to_remove = [c for c in train_dataset.column_names if c != 'labels']
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=cols_to_remove)
    val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=cols_to_remove)
    
    # 5. Freezing Logic
    # 5. Freezing Logic
    if "specter" in model_name:
        print(f"Configuring layers for SPECTER (Unfreeze last {Config.UNFREEZE_LAST_N_LAYERS})...")
        
        # 1. First, freeze EVERYTHING (Base Encoder)
        if hasattr(model, "bert"):
            encoder = model.bert.encoder
        elif hasattr(model, "roberta"):
            encoder = model.roberta.encoder
        else:
            encoder = model.base_model.encoder

        for param in encoder.parameters():
            param.requires_grad = False
            
        # 2. Always unfreeze the Classifier Head
        for param in model.classifier.parameters():
            param.requires_grad = True
            
        # 3. Smart Unfreezing of Encoder Layers
        if Config.UNFREEZE_LAST_N_LAYERS > 0:
            num_layers = len(encoder.layer) # Usually 12
            start_layer = num_layers - Config.UNFREEZE_LAST_N_LAYERS
            
            print(f"Unfreezing encoder layers {start_layer} to {num_layers - 1}...")
            for i in range(start_layer, num_layers):
                for param in encoder.layer[i].parameters():
                    param.requires_grad = True
    else:
        # BERT/RoBERTa Baselines: Full Fine-Tuning (Standard)
        print(f"Full fine-tuning for {model_name}...")
        
    # 6. Training
    # Data Slicing
    if Config.DATA_FRACTION < 1.0:
        subset_size = int(len(train_dataset) * Config.DATA_FRACTION)
        print(f"Subsampling Training Data: Using {subset_size} samples ({Config.DATA_FRACTION*100}%)")
        train_dataset = train_dataset.shuffle(seed=Config.SEED).select(range(subset_size))

    # 6. Training
    output_dir = Config.FINAL_MODEL_DIR
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=Config.EPOCHS_CLF,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE_CLF,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"./logs/{model_name}",
        logging_steps=10,
        seed=Config.SEED,
        fp16=(Config.DEVICE == "cuda"),
        use_cpu=(Config.DEVICE == "cpu"),
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    print(f"Saving final model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Finished training {model_name}.")

if __name__ == "__main__":
    # Test run
    # run_training("specter_tapt")
    pass
