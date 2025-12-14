# Specter 2 TAPT Classifier: The "Beat SciBERT" Project

## 🚀 Quick Start (New Workstation)
**Prerequisites**: Python 3.10+, CUDA-capable GPU (>16GB VRAM recommended for Phase 8).

1.  **Clone & Install**:
    ```bash
    git clone <repo_url>
    cd specter2-tapt-classifier
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Run the "Nuclear Option" (Phase 8)**:
    *   *Note: This requires significant GPU memory. If running on <16GB VRAM, reduce batch size in `src/phase8_setfit.py`.*
    ```bash
    python -m src.phase8_setfit
    ```

---

## 📖 Project Overview
The goal of this project was to build the ultimate scientific abstract classifier, starting with **SPECTER 2** and iteratively improving it to beat the domain-expert baseline, **SciBERT**.

## 🏆 Leaderboard (Final Results)

| Rank | Model | Strategy | Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **SciBERT** | **Domain Pre-training** | **98.08%** | **Reigning Champion** |
| 2 | Specter 2 | Full Fine-Tuning | 97.60% | Close Second |
| 3 | Ensemble | Synergy (Phase 6) | 97.12% | Robust but lower peak |
| 4 | DeBERTa-v3 | kNN Hybrid (Phase 7) | 96.15% | Underperformed |
| 5 | SetFit | Contrastive (Phase 8) | *Pending* | **Ready for Training** |

## 🧪 Development Phases

### Phase 1-3: The Foundation
- Established baselines.
- **Finding**: Unfrozen Specter 2 (97.60%) beat BERT and RoBERTa, proving the value of domain adaptation.

### Phase 4: The Shootout
- Benchmarked DeBERTa-v3 vs. SciBERT vs. BERT.
- **Finding**: SciBERT (98.08%) took the crown, proving that *pre-training on scientific text* > *modern architecture* for this specific task.

### Phase 5: Context Injection
- **Hypothesis**: Injecting synonyms and definitions into the text would help.
- **Result**: **Failure**. Accuracy dropped. The added noise confused the model more than the context helped.

### Phase 6: The Synergy Ensemble
- **Hypothesis**: Combining models would smooth out errors.
- **Result**: **97.12%**. Good, but the lower-performing models dragged down the average.

### Phase 7: The Cyborg (kNN + DeBERTa)
- **Hypothesis**: Using a kNN memory bank for inference would fix edge cases.
- **Result**: **96.15%**. No improvement over the base model.

### Phase 8: The Nuclear Option (SetFit)
- **Hypothesis**: Contrastive Learning can optimize the embedding geometry directly.
- **Status**: **Implemented & Verified**.
- **Note**: The pipeline is ready. Due to hardware constraints on the dev machine, the full training run is deferred to the high-performance workstation.

## 📂 Directory Structure
- `src/`: Source code for all phases.
- `data/`: Datasets (Raw and Processed).
- `models/`: Saved model artifacts (Ignored by Git).
- `results/plots/`: Visualizations from all phases.
- `results/logs/`: Execution logs.
- `docs/`: Implementation plans and notes.

## 🔧 Key Scripts
- `src/phase8_setfit.py`: The final contrastive learning implementation.
- `src/generate_augmented_data.py`: Data augmentation for Phase 8.
- `src/train_shootout.py`: Phase 4 benchmarking script.
