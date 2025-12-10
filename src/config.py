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
