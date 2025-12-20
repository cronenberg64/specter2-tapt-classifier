import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import os

# Config
DATA_PATH = "data/raw/scientific_abstracts_dataset.csv"
OUTPUT_DIR = "results/figures"
MODELS = {
    "BERT-Base (Baseline)": "google-bert/bert-base-uncased",
    "Specter-TAPT (Ph1)": "./models/specter_tapt_classifier"
}

# Common scientific words that act as "noise" across all domains
SCIENTIFIC_STOPWORDS = {
    'model', 'models', 'data', 'based', 'using', 'analysis', 'results', 'study', 
    'research', 'proposed', 'approach', 'results', 'performance', 'used', 'method',
    'different', 'system', 'problem', 'time', 'paper', 'new', 'show', 'provide',
    'information', 'framework', 'application', 'process', 'high', 'low', 'development'
}

def generate_tsne_clusters():
    print("--- Generating Phase 1 t-SNE Clusters ---")
    dataset = load_dataset("csv", data_files=DATA_PATH)["train"].train_test_split(test_size=0.2, seed=42)["test"]
    texts = dataset["text"]
    labels = dataset["label"]
    
    plt.figure(figsize=(18, 8))
    
    for i, (name, path) in enumerate(MODELS.items()):
        print(f"Processing {name}...")
        try:
            # Using SentenceTransformer with explicit pooling for raw HF models
            if not os.path.exists(path) and "google-bert" in path:
                # For baseline BERT, we use a standard SBERT model for reliable latent space comparison
                # since raw BERT-base CLS tokens are not great for t-SNE without pooling.
                model = SentenceTransformer("paraphrase-MiniLM-L6-v2") 
            else:
                model = SentenceTransformer(path)
                
            embeddings = model.encode(texts, show_progress_bar=True)
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto")
            X_2d = tsne.fit_transform(embeddings)
            
            plt.subplot(1, 2, i+1)
            sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=labels, palette="viridis", s=60, edgecolor="k", alpha=0.7)
            plt.title(f"{name}", fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.2)
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
            
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase1_baseline_clusters.png")
    print(f"Saved: {OUTPUT_DIR}/phase1_baseline_clusters.png")

def generate_keyword_analysis():
    print("--- Generating Phase 1 Jargon Analysis ---")
    df = pd.read_csv(DATA_PATH)
    
    # Custom Stopwords
    import nltk
    from nltk.corpus import stopwords
    try:
        nltk.download('stopwords')
        stop_words = list(stopwords.words('english')) + list(SCIENTIFIC_STOPWORDS)
    except:
        stop_words = list(SCIENTIFIC_STOPWORDS)

    # Stricter token pattern: Must start with a letter and contain at least one more character
    # This filters out artifacts like '_2', '10', etc.
    token_pattern = r"(?u)\b[a-zA-Z]\w+\b"
    
    tfidf = TfidfVectorizer(
        stop_words=stop_words, 
        max_features=2000, 
        ngram_range=(1, 1),
        token_pattern=token_pattern
    )
    
    plt.figure(figsize=(15, 8))
    domains = df['label'].unique()
    
    # Global TF-IDF to find globally frequent but domain-specific terms
    matrix = tfidf.fit_transform(df['text'])
    feature_names = tfidf.get_feature_names_out()
    
    for i, domain in enumerate(domains):
        domain_indices = df[df['label'] == domain].index
        domain_matrix = matrix[domain_indices]
        
        # Calculate mean TF-IDF score per word for this domain
        mean_scores = np.asarray(domain_matrix.mean(axis=0)).flatten()
        sorted_indices = mean_scores.argsort()[::-1]
        
        # Get top 10
        top_indices = sorted_indices[:10]
        top_terms = [feature_names[idx] for idx in top_indices]
        top_scores = [mean_scores[idx] for idx in top_indices]
        
        plt.subplot(1, 3, i+1)
        sns.barplot(x=top_scores, y=top_terms, palette="rocket")
        plt.title(f"{domain}", fontsize=14, fontweight='bold')
        
    plt.suptitle("Domain-Specific Jargon (Phase 1 Dataset)", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase1_domain_jargon.png")
    print(f"Saved: {OUTPUT_DIR}/phase1_domain_jargon.png")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    generate_tsne_clusters()
    generate_keyword_analysis()
