import torch

class Config:
    # Model ID
    MODEL_NAME = "allenai/specter2_base"
    
    # Paths
    DATA_PATH = "./data/raw/scientific_abstracts_dataset.csv"
    TAPT_OUTPUT_DIR = "./models/specter_tapt"
    FINAL_MODEL_DIR = "./models/specter_classifier"
    
    # NEW: Experiment Control Knobs (Defaults)
    USE_TAPT_WEIGHTS = True       # True = use ./models/specter_tapt, False = use allenai/specter2_base
    UNFREEZE_LAST_N_LAYERS = 0    # 0 = Freeze all. 2 = Unfreeze top 2. 12 = Unfreeze all.
    DATA_FRACTION = 0.2           # Reduced to 20% for CPU speed test
    
    # PHASE 4: THE GRANDMASTER
    PHASE4_DIR = "./models/phase4_shootout"
    MODEL_CANDIDATES = [
        "roberta-large",                    # The "Accuracy King" (Replaces DeBERTa due to Windows SP issues)
        "allenai/scibert_scivocab_uncased", # The "Domain Expert"
        "google-bert/bert-base-uncased"     # The Baseline (to re-confirm)
    ]
    EPOCHS_SHOOTOUT = 1      # Reduced for CPU speed
    LEARNING_RATE = 2e-5
    
    # Training Hyperparameters
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    LEARNING_RATE_TAPT = 2e-5      # Lower for TAPT to be gentle
    LEARNING_RATE_CLF = 5e-5       # Standard for classification
    EPOCHS_TAPT = 10               # Needs time to learn vocabulary
    EPOCHS_CLF = 5                 # Quick convergence expected
    
    # Hardware
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    SEED = 42
