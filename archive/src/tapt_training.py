import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
from src.config import Config
import os

def run_tapt():
    print(f"Loading data from {Config.DATA_PATH}...")
    df = pd.read_csv(Config.DATA_PATH)
    
    # For TAPT, we only need the text, labels don't matter (unsupervised)
    texts = df['text'].tolist()
    
    # Create a Hugging Face Dataset
    dataset = Dataset.from_dict({"text": texts})
    
    print(f"Loading tokenizer and model: {Config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(Config.MODEL_NAME, use_safetensors=True)
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=Config.MAX_LENGTH,
            return_special_tokens_mask=True
        )
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # FREEZING LAYERS 0-5
    # SPECTER 2 is based on BERT (usually). Let's inspect structure if needed, but standard is model.bert.encoder.layer
    # We want to freeze the bottom layers to preserve general citation knowledge
    print("Freezing layers 0-5 of the encoder...")
    
    # Verify the attribute path. For bert-base-uncased it is model.bert.encoder.layer
    # specter2_base is likely the same.
    if hasattr(model, "bert"):
        encoder_layers = model.bert.encoder.layer
    elif hasattr(model, "roberta"): # Just in case it's roberta based, though specter is bert
        encoder_layers = model.roberta.encoder.layer
    else:
        # Fallback or error if structure is unexpected. 
        # But we know specter is BERT-based.
        encoder_layers = model.bert.encoder.layer

    for i, layer in enumerate(encoder_layers):
        if i < 6: # Freeze 0, 1, 2, 3, 4, 5
            for param in layer.parameters():
                param.requires_grad = False
            print(f"  Layer {i} frozen.")
        else:
            print(f"  Layer {i} trainable.")
            
    # Data Collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
        output_dir=Config.TAPT_OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=Config.EPOCHS_TAPT,
        per_device_train_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE_TAPT,
        weight_decay=0.01,
        save_strategy="no", # We save at the end manually or use save_model
        logging_steps=10,
        seed=Config.SEED,
        fp16=(Config.DEVICE == "cuda"), # Use mixed precision if on GPU
        use_cpu=(Config.DEVICE == "cpu"), # Force CPU if specified
        push_to_hub=False,
        report_to="none" # Disable wandb etc for this simple script
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )
    
    print("Starting TAPT training...")
    trainer.train()
    
    print(f"Saving adapted model to {Config.TAPT_OUTPUT_DIR}...")
    trainer.save_model(Config.TAPT_OUTPUT_DIR)
    tokenizer.save_pretrained(Config.TAPT_OUTPUT_DIR)
    print("TAPT Complete.")

if __name__ == "__main__":
    run_tapt()
