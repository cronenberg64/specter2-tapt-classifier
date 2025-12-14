# Project Phase Results & Findings

This document tracks the scientific progress of the project, detailing the hypothesis, experiment, and conclusion for each development phase.

## Phase 1: Foundations & Data Pipeline
- **Goal**: Establish a robust data ingestion and processing pipeline.
- **Findings**:
    - The dataset consists of scientific abstracts from three distinct domains: Computer Science, Materials Science, and Medicine.
    - Data quality is high, but vocabulary is highly specialized.
- **Result**: Successfully implemented `collect_data.py` and created a clean CSV dataset.

## Phase 2: Task-Adaptive Pre-Training (TAPT)
- **Hypothesis**: Pre-training a language model on the *unlabeled* corpus of our specific domain will improve downstream performance.
- **Experiment**: Fine-tuned `allenai/specter2_base` on the raw abstract text (Masked Language Modeling).
- **Findings**:
    - The model loss decreased significantly during TAPT, indicating it successfully learned the domain vocabulary.
- **Result**: Produced a "scientific-literate" base model ready for classification.

## Phase 3: Initial Classifier Training
- **Hypothesis**: A simple classification head on top of the TAPT model will yield strong results.
- **Experiment**: Trained the TAPT-adapted model on the labeled dataset.
- **Findings**:
    - The model converged quickly.
    - Initial accuracy was high (>90%), validating the TAPT approach.
- **Result**: A working classifier that outperformed generic baselines.

## Phase 4: The "Grandmaster" Shootout
- **Hypothesis**: Newer architectures (DeBERTa) or domain-specific pre-training (SciBERT) will outperform the standard BERT baseline.
- **Experiment**: Trained and compared three models:
    1.  **DeBERTa-v3-base** (Generalist, high accuracy architecture)
    2.  **SciBERT** (Specialist, pre-trained on science)
    3.  **BERT-base** (Baseline)
- **Results**:
    | Model | Accuracy | Status |
    | :--- | :--- | :--- |
    | **SciBERT** | **98.08%** | **Winner** |
    | Ensemble | 97.12% | Runner-up |
    | BERT | 96.63% | Baseline |
    | DeBERTa | 96.15% | Underperformed |
- **Conclusion**: **Domain expertise beats architecture.** SciBERT's pre-training on scientific text provided a greater advantage than DeBERTa's advanced attention mechanisms for this specific task.

## Phase 5: Knowledge-Enhanced Classification (In Progress)
- **Hypothesis**: Injecting explicit domain keywords and synonyms into the input will bridge the vocabulary gap for generalist models like DeBERTa, potentially allowing them to surpass SciBERT.
- **Method**:
    1.  Build an **Associativity Map** using TF-IDF to find strong domain signals.
    2.  **Inject Context** (synonyms/domain hints) into the input text during training.
    3.  Retrain DeBERTa-v3.
- **Results**:
    - **Accuracy**: **97.12%**
    - **Comparison**:
        - vs. Baseline DeBERTa (96.15%): **+0.97% Improvement** (Success!)
        - vs. SciBERT (98.08%): **-0.96% Gap** (Failed to beat domain pre-training)
- **Conclusion**: Context injection **works**, significantly boosting the generalist model's performance. However, it is not enough to overcome the massive advantage of a model pre-trained entirely on scientific text (SciBERT). **SciBERT remains the champion.**
