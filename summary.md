# Project Summary: SPECTER2 TAPT Classifier

This document summarizes the development lifecycle of the Scientific Abstract Classifier project, detailing the progression from initial setup to the final "Nuclear Option".

## Project Goal
To build a high-accuracy text classifier for scientific abstracts, leveraging Task-Adaptive Pre-Training (TAPT), Ensembling, and Contrastive Learning.

## Development Phases

### Phase 1-3: Foundation & Setup
- **Objective**: Establish baselines and test Specter 2.
- **Result**: Unfrozen Specter 2 achieved **97.60%**, beating standard BERT.

### Phase 4: The "Grandmaster" Shootout
- **Objective**: Benchmark top models.
- **Result**: **SciBERT** won with **98.08%**.

### Phase 5: Contextual Injection
- **Objective**: Improve performance by injecting synonyms and definitions.
- **Result**: Performance degraded. The added text introduced noise.

### Phase 6: The Synergy Ensemble
- **Objective**: Combine predictions from multiple models.
- **Result**: **97.12%**. Robust, but failed to beat the single best model (SciBERT).

### Phase 7: The Cyborg (kNN-DeBERTa)
- **Objective**: Augment DeBERTa with a k-Nearest Neighbors memory bank.
- **Result**: **96.15%**. The kNN component did not add value over the base classifier.

### Phase 8: The Nuclear Option (SetFit)
- **Objective**: Use Contrastive Learning (SetFit) to optimize embedding geometry.
- **Status**: **Ready for Training**.
- **Details**: The pipeline uses `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (safetensors) and `CosineSimilarityLoss`. Full training requires >16GB VRAM.

## Conclusion
**SciBERT (98.08%)** remains the reigning champion. While modern architectures (DeBERTa) and complex techniques (Ensembling, kNN) were tested, the domain-specific pre-training of SciBERT proved unbeatable on this dataset. Phase 8 (SetFit) offers a final theoretical avenue for improvement, pending a full training run on high-end hardware.
