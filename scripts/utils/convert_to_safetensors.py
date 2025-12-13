from transformers import AutoModel, AutoTokenizer
import torch

model_id = "allenai/specter2_base"
local_path = "./models/specter2_base_safetensors"

print(f"Downloading {model_id}...")
# Load with standard transformers (which we know works with use_safetensors=False if we don't force it, wait, does it?)
# The error came from SetFit -> SentenceTransformer -> torch.load.
# Transformers library usually handles this better or allows it with a warning.
# Let's try to load with AutoModel and save as safetensors.

model = AutoModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Saving to {local_path} with safetensors...")
model.save_pretrained(local_path, safe_serialization=True)
tokenizer.save_pretrained(local_path)

print("Conversion complete.")
