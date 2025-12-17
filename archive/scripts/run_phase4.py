from src.train_shootout import run_shootout
from src.visualize_embeddings import run_visualization
from src.ensemble_scoring import run_ensemble

def main():
    # Step 1: Train the new candidates
    run_shootout()
    
    # Step 2: Generate the pretty plots
    run_visualization()
    
    # Step 3: Run the combined scoring
    run_ensemble()

if __name__ == "__main__":
    main()
