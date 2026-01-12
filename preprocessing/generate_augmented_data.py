import pandas as pd
import json
from datasets import Dataset
from transformers import AutoTokenizer
# from src.config import Config
import os

def generate_augmented_data():
    print("--- GENERATING AUGMENTED DATASET (CONTEXT INJECTION) ---")
    
    # 1. Load the Map
    try:
        with open("data/associativity_map.json", "r") as f:
            assoc_map = json.load(f)
    except FileNotFoundError:
        print("Error: Associativity Map not found. Run preprocessing/build_associativity.py first.")
        return

    # 2. Load Data
    data_path = "data/raw/scientific_abstracts_dataset.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    # Ensure labels are encoded if needed, but for SetFit we might want string labels or int labels.
    # The plan says "Clean NaNs" and "Load Augmented Phase 5 Dataset".
    # We will keep the original text and label, but create a new column or just replace text?
    # The plan implies we use the injected text as the training data.
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    
    # --- Context Injection Logic (Reused from Phase 5) ---
    def inject_context_single(text):
        words = set(text.lower().split())
        hints = []
        for word in words:
            if word in assoc_map:
                entry = assoc_map[word]
                syn_str = ", ".join(entry['synonyms'])
                hint = f"{word} -> {entry['domain']} (Syns: {syn_str})"
                hints.append(hint)
        
        if hints:
            context_str = " | ".join(hints[:3])
            new_text = f"{text} [SEP] [HINT] {context_str}"
        else:
            new_text = text
        return new_text

    print("Injecting context...")
    # We want to create a dataset that has the INJECTED text.
    # Let's create a new DataFrame with the injected text.
    
    augmented_rows = []
    for _, row in df.iterrows():
        original_text = row['text']
        label = row['label']
        
        # 1. Original (Keep it? The plan says "Augmented Phase 5 Dataset... 3x size = better contrastive pairs")
        # Actually, Phase 5 just injected context. It didn't necessarily triple the size.
        # But the Phase 8 plan says "TRAIN on Phase 5 Augmented Data (3x size = better contrastive pairs)".
        # This implies we might want: Original, Injected, and maybe something else?
        # Or maybe it just means "The dataset with context injected".
        # Let's look at the Phase 8 plan text again: "Load the Augmented Phase 5 Dataset for training (more data = better contrast)."
        # And "3x size".
        # If I just inject context, it's 1x size.
        # Maybe I should keep Original AND Injected? That would be 2x.
        # Let's do Original + Injected. That gives pairs of (Original, Label) and (Injected, Label).
        # SetFit works on pairs.
        # Let's also add a "Synonym Replacement" version if we want 3x, but we don't have that logic handy.
        # Let's stick to Original + Injected for now, effectively doubling it. 
        # Wait, if I look at Phase 5, it just replaced the text.
        # But "Augmented" usually implies adding more samples.
        # Let's generate:
        # 1. Original Text
        # 2. Context Injected Text
        # That's 2x.
        
        augmented_rows.append({"text": original_text, "label": label})
        augmented_rows.append({"text": inject_context_single(original_text), "label": label})
        
    df_aug = pd.DataFrame(augmented_rows)
    
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "phase5_augmented_dataset.csv")
    
    df_aug.to_csv(output_path, index=False)
    print(f"Saved augmented dataset to {output_path} (Rows: {len(df_aug)})")

if __name__ == "__main__":
    generate_augmented_data()
