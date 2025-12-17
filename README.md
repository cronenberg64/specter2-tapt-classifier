# SciBERT vs. Generalist Giants: A Contrastive Learning Benchmark

## Objective
Investigating whether modern **Generalist Architectures** (BERT-Large, 340M params) can outperform **Domain-Specific Models** (SciBERT, 110M params) on scientific text classification when optimized with **Contrastive Learning**.

## Methodology
* **Pipeline:** Sentence Transformers with Batch Hard Triplet Loss.
* **Data:** 1,000 Scientific Abstracts (Augmented to ~2,000 samples).
* **Leakage Prevention:** Rigorous filtering to ensure no test-set synonyms leaked into training.
* **Hardware:** Trained on NVIDIA RTX A4000 (16GB VRAM).

## Key Results
Despite being 3x smaller, the domain-specific model won.

| Model | Parameters | Training Time | Accuracy |
| :--- | :--- | :--- | :--- |
| **SciBERT** | 110M | ~22 sec | **99.52%** |
| BERT-Large | 340M | ~35 min | 98.56% |

**Conclusion:** Domain pre-training provides better conceptual separation for scientific jargon than raw parameter scale.

## Visualization
![Latent Space Visualization](results/figures/cluster_compare_final.png)
*Left: SciBERT shows distinct, dense clusters. Right: BERT-Large shows slightly fuzzier boundaries.*

## How to Run
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the experiment:
   `python main.py`
3. Visualize results:
   `python visualize.py`
