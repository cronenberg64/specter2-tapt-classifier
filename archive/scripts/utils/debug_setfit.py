from sentence_transformers import SentenceTransformer
try:
    print("Attempting to load SentenceTransformer with use_safetensors=True...")
    model = SentenceTransformer("allenai/specter2_base", model_kwargs={"use_safetensors": True})
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")

try:
    print("Attempting to load SentenceTransformer with trust_remote_code=True...")
    model = SentenceTransformer("allenai/specter2_base", trust_remote_code=True)
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
