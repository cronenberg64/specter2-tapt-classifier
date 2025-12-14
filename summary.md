# Project Summary: SPECTER2 TAPT Classifier

This document summarizes the development lifecycle of the Scientific Abstract Classifier project, detailing the progression from initial setup to the final "Grandmaster" model shootout.

## Project Goal
To build a high-accuracy text classifier for scientific abstracts, leveraging Task-Adaptive Pre-Training (TAPT) and comparing state-of-the-art NLP models.

## Development Phases

### Phase 1: Foundation & Setup
- **Objective**: Establish the project structure and data pipeline.
- **Key Actions**:
    - Set up the Python environment and dependencies.
    - Implemented data collection scripts (`src/collect_data.py`).
    - Created the initial configuration (`src/config.py`).

### Phase 2: Task-Adaptive Pre-Training (TAPT)
- **Objective**: Adapt the `allenai/specter2_base` model to our specific scientific domain.
- **Key Actions**:
    - Implemented `src/tapt_training.py` to fine-tune the language model on the unlabeled corpus.
    - This step allowed the model to learn the specific vocabulary and patterns of the dataset before classification training.

### Phase 3: Classifier Training
- **Objective**: Train a classification head on top of the pre-trained body.
- **Key Actions**:
    - Implemented `src/classifier_training.py`.
    - Fine-tuned the model on the labeled dataset.
    - Achieved strong initial performance, validating the TAPT approach.

### Phase 4: The "Grandmaster" Shootout
- **Objective**: Push for maximum accuracy by benchmarking top models against each other.
- **Candidates**:
    1.  **DeBERTa-v3-base**: The modern accuracy king.
    2.  **SciBERT**: The domain expert (pre-trained on scientific text).
    3.  **BERT-base**: The reliable baseline.
- **Key Actions**:
    - Implemented `src/train_shootout.py` to train all candidates.
    - Created `src/visualize_embeddings.py` to visualize cluster separation.
    - Developed `src/ensemble_scoring.py` to combine models.

## Final Results

The Phase 4 shootout yielded the following accuracy scores:

| Rank | Model | Accuracy | Insight |
| :--- | :--- | :--- | :--- |
| **1** | **SciBERT** | **98.08%** | **The Winner.** Domain pre-training proved more valuable than newer architecture. |
| 2 | Ensemble | 97.12% | Good, but weighed down by the lower-performing model. |
| 3 | BERT | 96.63% | A very strong baseline. |
| 4 | DeBERTa | 96.15% | Surprisingly underperformed on this specific task. |

## Conclusion
The project successfully demonstrated that for specialized scientific text classification, **SciBERT** is the optimal choice, achieving a production-grade accuracy of **98.08%**.
