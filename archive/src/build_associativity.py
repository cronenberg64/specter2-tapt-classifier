import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import nltk
from nltk.corpus import wordnet
import sys
import os
# Add project root to sys.path to allow running script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure we have the dictionary downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_synonyms(word):
    """Fetches top 3 unique synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            clean_lemma = lemma.name().replace("_", " ")
            if clean_lemma.lower() != word.lower():
                synonyms.add(clean_lemma)
    return list(synonyms)[:3]

def build_associativity_map():
    print("--- BUILDING DOMAIN ASSOCIATIVITY MAP (TF-IDF) ---")
    
    # 1. Load Data
    df = pd.read_csv("data/raw/scientific_abstracts_dataset.csv")
    
    # 2. Group text by Domain (Label) to find domain-specific jargon
    domain_text = df.groupby('label')['text'].apply(lambda x: " ".join(x)).reset_index()
    
    # 3. Compute TF-IDF
    # We ignore common English words ('stop_words') to find unique jargon
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    matrix = tfidf.fit_transform(domain_text['text'])
    feature_names = np.array(tfidf.get_feature_names_out())
    
    associativity_map = {}
    
    print("Extracting strong domain keywords...")
    for i, row in domain_text.iterrows():
        domain = row['label']
        
        # Get the highest scoring words for this domain
        domain_scores = matrix[i].toarray()[0]
        top_indices = domain_scores.argsort()[-60:][::-1] # Top 60 words
        
        top_words = feature_names[top_indices]
        
        for word in top_words:
            # Only add if not already assigned to another domain (or overwrite if score is higher)
            if word not in associativity_map:
                associativity_map[word] = {
                    "domain": domain,
                    "synonyms": get_synonyms(word)
                }
            
    # 4. Save the Map
    # Need to import Config here or just use absolute path logic if Config is not imported.
    # Since Config is not imported in the original script, let's just use os.path logic or import Config.
    # Importing Config is cleaner.
    from src.config import Config
    output_path = Config.ASSOCIATIVITY_MAP_PATH
    with open(output_path, "w") as f:
        json.dump(associativity_map, f, indent=4)
        
    print(f"Map built with {len(associativity_map)} keyword associations.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    build_associativity_map()
