import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.train_shootout import run_shootout
from experiments.visualize_embeddings import run_visualization
from experiments.ensemble_scoring import run_ensemble

def main():
    print("=========================================")
    print("   PHASE 4: THE GRANDMASTER PROTOCOL")
    print("=========================================")
    
    # Step 1: Train the new candidates
    print("\n[STEP 1] The Shootout: Training RoBERTa-Large, SciBERT, BERT...")
    run_shootout()
    
    # Step 2: Generate the pretty plots
    print("\n[STEP 2] The Visualizer: Generating t-SNE plots...")
    run_visualization()
    
    # Step 3: Run the combined scoring
    print("\n[STEP 3] The Ensemble: Fusing Models...")
    run_ensemble()
    
    print("\n>>> PHASE 4 COMPLETE. <<<")

if __name__ == "__main__":
    main()
