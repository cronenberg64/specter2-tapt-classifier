# Project Summary: Specter 2 TAPT Classifier

## 1. Project Goal
The objective of this project was to build a text classifier for scientific abstracts using **SPECTER 2** with **Task-Adaptive Pre-Training (TAPT)** and a **Layer Freezing** strategy. The performance was compared against standard baselines (**BERT** and **RoBERTa**).

## 2. Methodology

### Data Acquisition
- **Source**: arXiv API.
- **Categories**:
    - `q-bio.NC` (Neuroscience)
    - `cond-mat.mtrl-sci` (Materials Science)
    - `q-bio.QM` (Bioinformatics)
- **Dataset Size**: Approximately 1,050 unique abstracts (balanced classes).

### TAPT (Task-Adaptive Pre-Training)
- **Model**: `allenai/specter2_base`.
- **Objective**: Masked Language Modeling (MLM).
- **Strategy**: Continued pre-training on the domain-specific corpus for 10 epochs.
- **Regularization**: Layers 0-5 of the encoder were frozen to preserve general citation knowledge.

### Classification Fine-Tuning
- **Models Compared**:
    1.  **Specter TAPT**: Pre-trained on domain data, encoder frozen, classifier head trained.
    2.  **BERT (`bert-base-uncased`)**: Standard full fine-tuning.
    3.  **RoBERTa (`roberta-base`)**: Standard full fine-tuning.
- **Training Config**: 5 epochs, batch size 8, learning rate 5e-5.

## 3. Implementation Details
The project is structured as follows:
- `src/collect_data.py`: Fetches and cleans data from arXiv.
- `src/tapt_training.py`: Performs TAPT on Specter 2.
- `src/classifier_training.py`: Handles training for all three models.
- `src/evaluation.py`: Generates metrics (Accuracy, F1) and confusion matrices.
- `run_pipeline.py`: Orchestrates the entire workflow.

## 4. Results
The models were evaluated on a held-out test set (20%).

| Model | Accuracy | F1 Score |
| :--- | :--- | :--- |
| **BERT** | **0.9760** | **0.9759** |
| **RoBERTa** | 0.9663 | 0.9662 |
| **Specter TAPT** | 0.9471 | 0.9468 |

**Analysis**:
- **BERT** achieved the highest performance.
- **Specter TAPT** performed slightly lower, likely due to the **frozen encoder** strategy. While this preserves citation embeddings, it restricts the model's ability to adapt deep representations to the specific classification task compared to the fully fine-tuned baselines.

## 5. Challenges & Solutions
During implementation, several technical issues were resolved:
1.  **PyTorch DLL Error**: Initial installation failed on Windows.
    - *Fix*: Reinstalled PyTorch with CUDA 12.1 support (`pip install torch ... --index-url .../cu121`).
2.  **`torch.load` Vulnerability**: `transformers` blocked loading unsafe files.
    - *Fix*: Enforced `use_safetensors=True` in model loading calls.
3.  **Data Collation Error**: `Trainer` failed with string columns.
    - *Fix*: Renamed target column to `labels` and removed non-numeric columns (`text`, `label`) before passing to the trainer.

## 6. Conclusion
The project successfully demonstrated the TAPT pipeline. While the frozen Specter model didn't beat the fully fine-tuned baselines in this specific low-data regime, the pipeline is robust and ready for further experimentation (e.g., unfreezing layers, larger datasets).
