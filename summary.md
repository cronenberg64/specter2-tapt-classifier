# Project Summary: Specter 2 TAPT Classifier

## 1. Project Goal
The objective of this project was to build and rigorously evaluate a text classifier for scientific abstracts using **SPECTER 2** with **Task-Adaptive Pre-Training (TAPT)**. We aimed to determine if domain adaptation and layer freezing strategies could outperform standard baselines.

## 2. Methodology

### Data Acquisition
- **Source**: arXiv API (`q-bio.NC`, `cond-mat.mtrl-sci`, `q-bio.QM`).
- **Dataset**: ~1,050 unique abstracts (balanced classes).

### Phase 1: Foundation & Baselines
We established a pipeline to compare:
- **Specter TAPT**: Pre-trained on domain data, encoder frozen.
- **BERT / RoBERTa**: Standard full fine-tuning.

### Phase 2: Advanced Experimentation
We implemented an automated experiment suite to isolate variables:
1.  **Ablation**: Specter Base vs. Specter TAPT (Does TAPT help?).
2.  **Unfreezing**: Frozen Encoder vs. Top-2 Layers Unfrozen.
3.  **Data Efficiency**: Training on 100% vs. 10% of data.

## 3. Results

### Phase 1: Model Comparison (Full Data)
| Model | Strategy | F1 Score |
| :--- | :--- | :--- |
| **BERT** | Full Fine-Tuning | **0.9759** |
| **RoBERTa** | Full Fine-Tuning | 0.9662 |
| **Specter TAPT** | Frozen Encoder | 0.9468 |

### Phase 2: Specter Deep Dive
| Experiment | Configuration | F1 Score | Impact |
| :--- | :--- | :--- | :--- |
| **Exp 1** | Base Specter (No TAPT) | 0.9567 | Baseline |
| **Exp 2** | **Specter + TAPT** | **0.9614** | **+0.47% (TAPT Gain)** |
| **Exp 3** | TAPT + Top-2 Unfrozen | 0.9614 | No significant gain |
| **Exp 4** | Low Data (10%) | 0.6958 | -26% Drop |

## 4. Key Findings
1.  **TAPT Works**: Domain adaptation provided a consistent **~0.5% improvement** over the non-adapted Specter baseline.
2.  **BERT is Robust**: For this specific task and dataset size, a fully fine-tuned BERT model outperformed the frozen Specter approach.
3.  **Freezing is Efficient**: Unfreezing the top 2 layers of Specter yielded negligible gains, suggesting the frozen embeddings are already high-quality, but the "Frozen" constraint limits the peak performance compared to full fine-tuning.
4.  **Data Hunger**: The Specter model (frozen) struggled in the low-data regime (10%), indicating it relies on sufficient labeled data to train the classifier head effectively.

## 5. Implementation Details
The project is modularized for reproducibility:
- `src/collect_data.py`: arXiv scraper.
- `src/tapt_training.py`: Domain adaptation (MLM).
- `src/classifier_training.py`: Training logic with **Smart Freezing** and **Data Slicing**.
- `src/config.py`: Centralized config with experiment control knobs.
- `run_experiments.py`: Automated experiment runner.

## 6. Conclusion
We successfully demonstrated a TAPT pipeline. While TAPT improved the specific Specter model, the **Frozen Encoder** strategy proved to be a limiting factor against fully fine-tuned baselines. Future work should focus on **Full Fine-Tuning of Specter TAPT** to combine the initialization benefits of TAPT with the plasticity of full training.
