import sentencepiece as spm
print("Imported sentencepiece")
try:
    s = spm.SentencePieceProcessor()
    print("Created processor")
except Exception as e:
    print(f"Error: {e}")
