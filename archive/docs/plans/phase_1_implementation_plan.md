This plan implements the **SPECTER 2 + TAPT + Layer Freezing** strategy

-----

### **Part 1: The File Structure**

**Instruction to Agent:** "Initialize a Python project with the following directory structure."

```text
AI_Final_Project/
├── data/
│   ├── raw/                   # Where the arXiv script saves data
│   └── processed/             # (Optional) If you save train/test splits
├── models/
│   ├── specter_tapt/          # Output of Step 1 (Domain Adapted Model)
│   └── specter_classifier/    # Output of Step 2 (Final Fine-Tuned Model)
├── src/
│   ├── config.py              # Hyperparameters (Epochs, Batch Size)
│   ├── collect_data.py        # The arXiv scraper
│   ├── tapt_training.py       # Phase 1: Unsupervised Domain Adaptation
│   ├── classifier_training.py # Phase 2: Supervised Fine-Tuning
│   └── evaluation.py          # Metrics & Confusion Matrix
├── run_pipeline.py            # Main execution script
└── requirements.txt           # Dependencies
```

-----

### **Part 2: The Dependencies**

**Instruction to Agent:** "Create a `requirements.txt` file with these specific libraries to ensure compatibility with SPECTER 2."

```text
torch>=2.0.0
transformers>=4.35.0
scikit-learn
pandas
numpy
accelerate>=0.21.0
datasets
tqdm
matplotlib
seaborn
```

-----

### **Part 3: The Code Implementation Steps**

#### **Step 1: Data Acquisition (`src/collect_data.py`)**

**Context:** We need exactly 3 classes to match the assignment options. We use arXiv categories: `q-bio.NC` (Neuroscience), `cond-mat.mtrl-sci` (Materials Science), and `q-bio.QM` (Bioinformatics).

**Code for Agent:**
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import time
import random

# CONFIGURATION
# We fetch slightly more than 300 to account for potential duplicates or empty abstracts
SAMPLES_PER_CLASS = 350 

# arXiv API Categories mapping
# See: https://arxiv.org/category_taxonomy
TOPICS = {
    "Bioinformatics": "cat:q-bio.QM",       # Quantitative Methods (often used for Bioinformatics)
    "Neuroscience": "cat:q-bio.NC",         # Neurons and Cognition
    "Materials Science": "cat:cond-mat.mtrl-sci" # Materials Science
}

def fetch_arxiv_data(topic_name, query_code, max_results):
    """
    Fetches raw XML from arXiv API and parses titles/abstracts.
    """
    base_url = 'http://export.arxiv.org/api/query?'
    # Sort by 'submittedDate' ensures you get a somewhat random/recent slice 
    # rather than the same top 300 papers everyone else gets.
    query = f"search_query={query_code}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    
    print(f"Fetching {max_results} papers for {topic_name}...")
    
    try:
        response = urllib.request.urlopen(base_url + query).read()
        root = ET.fromstring(response)
        
        # arXiv XML namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text
            summary = entry.find('atom:summary', ns).text
            
            # CLEANING: arXiv abstracts have lots of \n newlines. Flatten them.
            clean_summary = summary.replace('\n', ' ').strip()
            clean_title = title.replace('\n', ' ').strip()
            
            if len(clean_summary) < 50: # Skip broken/empty abstracts
                continue
                
            papers.append({
                "label": topic_name,
                "text": clean_summary, # Abstract only (as per your strict constraint)
                "title_metadata": clean_title # Keep metadata just in case, but won't train on it
            })
            
        return papers
        
    except Exception as e:
        print(f"Error fetching {topic_name}: {e}")
        return []

# MAIN EXECUTION
all_data = []

for topic, code in TOPICS.items():
    data = fetch_arxiv_data(topic, code, SAMPLES_PER_CLASS)
    all_data.extend(data)
    # Politeness delay to respect arXiv server rules
    time.sleep(3) 

# Create DataFrame
df = pd.DataFrame(all_data)

# FINAL CLEANUP
# 1. Remove duplicates (crucial for "Unique Dataset" rule)
df.drop_duplicates(subset=['text'], inplace=True)

# 2. Shuffle data (so classes aren't ordered 1-2-3)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Check balance
print("\nDataset Balance:")
print(df['label'].value_counts())

# 4. Save
filename = "scientific_abstracts_dataset.csv"
df.to_csv(filename, index=False)
print(f"\nSuccess! Saved {len(df)} abstracts to {filename}")


#### **Step 2: Configuration (`src/config.py`)**

**Instruction to Agent:** "Create a config file to centralize our hyperparameters. This allows us to easily toggle TAPT on/off for experimentation."

```python
import torch

class Config:
    # Model ID
    MODEL_NAME = "allenai/specter2_base"
    
    # Paths
    DATA_PATH = "./data/raw/scientific_abstracts_dataset.csv"
    TAPT_OUTPUT_DIR = "./models/specter_tapt"
    FINAL_MODEL_DIR = "./models/specter_classifier"
    
    # Training Hyperparameters
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    LEARNING_RATE_TAPT = 2e-5      # Lower for TAPT to be gentle
    LEARNING_RATE_CLF = 5e-5       # Standard for classification
    EPOCHS_TAPT = 10               # Needs time to learn vocabulary
    EPOCHS_CLF = 5                 # Quick convergence expected
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
```

#### **Step 3: The Innovation - TAPT (`src/tapt_training.py`)**

**Instruction to Agent:** "Create a script that performs Task-Adaptive Pre-Training (TAPT). It must:

1.  Load `AutoModelForMaskedLM` with `allenai/specter2_base`.
2.  **Freeze layers 0-5** of the encoder to preserve citation knowledge.
3.  Train on *all* abstracts (ignoring labels) using Masked Language Modeling.
4.  Save the adapted encoder."

<!-- end list -->

```python
# [Snippet for Agent Reference]
# Setup freezing logic
for param in model.bert.encoder.layer[:6].parameters():
    param.requires_grad = False
    
# Use DataCollatorForLanguageModeling(mlm_probability=0.15)
# Train using the 'text' column of the CSV
```

#### **Step 4: The Fine-Tuning - Classification (`src/classifier_training.py`)**

**Instruction to Agent:** "Create the classification training script. It must be flexible to train different models."

1.  Accept a `model_name` argument.
2.  If `model_name` is "specter_tapt", load from `Config.TAPT_OUTPUT_DIR`.
3.  If `model_name` is "bert", load `bert-base-uncased`.
4.  If `model_name` is "roberta", load `roberta-base`.
5.  **Freeze layers:**
    *   For SPECTER: Freeze encoder (layers 0-11).
    *   For BERT/RoBERTa: Fine-tune all layers (standard practice for baselines) OR freeze if you want a direct "feature extraction" comparison. *Recommendation: Fine-tune all layers for baselines to give them a fair chance.*
6.  Use `StratifiedShuffleSplit`.
7.  Save each model to `./models/{model_name}_classifier`.

#### **Step 5: Evaluation & Visualization (`src/evaluation.py`)**

**Instruction to Agent:** "Create a function that loads ALL trained models and compares them."

1.  Loop through: `['specter_tapt', 'bert', 'roberta']`.
2.  Load each model and run inference on the Test Set.
3.  Generate a **Combined Bar Chart** comparing F1-scores.
4.  Generate individual Confusion Matrices.
5.  Save results to `results/comparison_report.csv`.

-----

### **Part 4: The Execution Script (`run_pipeline.py`)**

**Instruction to Agent:** "Orchestrate the full comparison."

```python
from src.collect_data import fetch_and_save_data
from src.tapt_training import run_tapt
from src.classifier_training import run_training
from src.evaluation import evaluate_all_models
import os

def main():
    # 1. Data
    if not os.path.exists("./data/raw/scientific_abstracts_dataset.csv"):
        fetch_and_save_data()
    
    # 2. TAPT (Only for SPECTER)
    # Check if TAPT is already done to save time
    if not os.path.exists("./models/specter_tapt"):
        run_tapt()
    
    # 3. Train Classifiers
    models_to_train = ["specter_tapt", "bert", "roberta"]
    for model in models_to_train:
        print(f"--- Training {model} ---")
        run_training(model_name=model)
    
    # 4. Evaluation
    evaluate_all_models()

if __name__ == "__main__":
    main()
```

-----

### **Part 5: Writing the Report (The Narrative)**

When the code finishes, you will have the logs and the confusion matrix. Here is how you map the **code** to the **report sections**:

**1. Data Acquisition Section:**

> "We utilized the `collect_data.py` module to query the arXiv API for categories `q-bio.NC`, `cond-mat.mtrl-sci`, and `q-bio.QM`. This ensured a balanced dataset of N=1050 unique abstracts."

**2. Methodology Section (The "Frontier" Part):**

> "We implemented a **Task-Adaptive Pre-Training (TAPT)** pipeline (`src/tapt_training.py`). By continuing the Masked Language Modeling (MLM) objective on our corpus for 10 epochs with a learning rate of 2e-5, we adapted the SPECTER 2 vocabulary to our specific scientific sub-domains."

**3. Model Architecture Section:**

> "To prevent catastrophic forgetting of the citation-based embeddings, we utilized a **Layer Freezing** strategy (`src/classifier_training.py`). We froze the bottom 6 layers during TAPT and the entire encoder during classification, training only the classification head. This leverages the pre-trained citation graph structure of SPECTER 2 while minimizing the risk of overfitting on a small dataset."
