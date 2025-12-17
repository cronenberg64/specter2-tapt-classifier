from src.phase6_ensemble import run_synergy_ensemble
from src.visualize_phase5 import run_visualization as run_viz_p5
from src.visualize_phase6 import run_phase6_viz

def main():
    # 1. Run Ensemble Logic
    run_synergy_ensemble()
    
    # 2. Run Visualizations
    run_viz_p5()
    run_phase6_viz()

if __name__ == "__main__":
    main()
