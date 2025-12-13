from transformers import AutoTokenizer
import sys

model_name = "microsoft/deberta-v3-base"
print(f"Loading tokenizer for {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
