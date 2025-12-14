from src.build_associativity import build_associativity_map
from src.train_contextual import run_contextual_training
import os

def main():
    # Step 1: Build the knowledge base
    if not os.path.exists("data/associativity_map.json"):
        build_associativity_map()
    else:
        print("Associativity Map found. Skipping build.")
        
    # Step 2: Train the Enhanced Model
    run_contextual_training()

if __name__ == "__main__":
    main()
