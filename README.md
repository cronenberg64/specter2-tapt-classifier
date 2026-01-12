# Scientific Abstract Classifier using a SetFit and Contrastive Learning Fine-tuned Language Model

This repo is made for the completion of the final project for the Artificial Intelligence class. This includes all the past experiments that helped lead me to the final model but was ultimately not used or is of no relevance to the final submission, they are merely for reference and for my personal documentation/tracking.

## Results

The model was trained twice to reduce error and both runs produced very similar performance metrics.

### Run 1 - Confusion Matrix
Shows the classification accuracy across all three scientific domains (Bioinformatics, Neuroscience, Materials Science).

![Confusion Matrix - Run 1](results/figures/final_confusion_matrix.png)

### Run 1 - Class Performance Metrics
Comparison of Precision, Recall, and F1-Score for each scientific domain.

![Class Performance - Run 1](results/figures/final_class_performance.png)

### Run 1 - t-SNE Embedding Visualization
2D visualization of how the model separates the three scientific domains in its learned embedding space.

![t-SNE Clusters - Run 1](results/figures/final_tsne_clusters.png)

---

### Run 2 - Confusion Matrix
Second training run showing consistent classification performance.

![Confusion Matrix - Run 2](results/figures/final_confusion_matrix_1.png)

### Run 2 - Class Performance Metrics
Second run metrics confirming reproducible results.

![Class Performance - Run 2](results/figures/final_class_performance_1.png)

### Run 2 - t-SNE Embedding Visualization
Second run showing similar cluster separation patterns.

![t-SNE Clusters - Run 2](results/figures/final_tsne_clusters_1.png)

**Note:** The consistency between both runs demonstrates the robustness of the SetFit contrastive learning approach and validates the reliability of the model's performance metrics.

---

### Model Comparisons

#### SetFit with Different Base Models
Comparison of SciBERT, BERT-Large, and BERT-base using the SetFit contrastive learning framework.

![SetFit Model Comparison](results/figures/final_comparison.png)

SciBERT (99.52%) significantly outperforms BERT-Large (98.94%) and BERT-base (93.75%), demonstrating the value of domain-specific pre-training.

#### SetFit vs Standard Fine-tuning
Comparison of standard fine-tuning approaches across different transformer models.

![Fine-tuning Comparison](results/figures/finetuning_comparison.png)

The SetFit contrastive learning approach produces tighter cluster separations compared to traditional fine-tuning methods.

## Quick Start (Running the Classifier)
If you only want to run the final classifier verification and training:

1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Main Script**:
    ```bash
    python JonathanSetiawan_aifinal.py
    ```
    *This will train the SetFit model on the augmented dataset, evaluate it, and generate results in `results/figures/` assuming that the dataset is located correctly.*

---

## Dataset Creation (Step-by-Step)
If you wish to reproduce the entire dataset creation process from scratch, follow these steps:

### 0. Collect Raw Data from arXiv (Optional)
If you need to fetch fresh abstracts from the arXiv API, run the data collection script. This queries arXiv for scientific abstracts across three domains: Bioinformatics, Neuroscience, and Materials Science.
```bash
python archive/src/collect_data.py
```
*Output: `data/raw/scientific_abstracts_dataset.csv` (~1,037 abstracts)*

> **Note:** The raw dataset is already included in the repository. Only run this step if you want to collect a fresh set of abstracts from arXiv.

### 1. Build the Associativity Map
This script analyzes the raw scientific abstracts to find domain-specific keywords and synonyms using TF-IDF.
```bash
python preprocessing/build_associativity.py
```
*Output: `data/associativity_map.json`*

### 2. Generate Augmented Dataset
This script uses the map to inject context hints into the raw abstracts, creating the final dataset.
```bash
python preprocessing/generate_augmented_data.py
```
*Output: `data/processed/phase5_augmented_dataset.csv` (Note: The main script expects `augmented_dataset.csv`, so you may need to rename it or update the script if you regenerate it).*

---

## Project Structure
*   `JonathanSetiawan_aifinal.py`: Main execution script for training and evaluation.
*   `data/`: Contains raw and processed datasets.
*   `preprocessing/`: Scripts used to create and augment the dataset.
*   `archive/`: Contains previous experimental code and auxiliary files.
*   `models/`: Stores the trained final model.
*   `results/figures/`: Contains sample output figures from a previous run. These will be overwritten when you run the main script.

## Requirements
Ensure you have the dependencies listed in `requirements.txt` installed. Key libraries include:
*   `setfit`
*   `pandas`
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`
*   `nltk`
