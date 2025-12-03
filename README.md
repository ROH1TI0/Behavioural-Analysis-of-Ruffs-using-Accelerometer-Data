# Behavioural-Analysis-of-Ruffs-using-Accelerometer-Data

**💻 Project for CSCI 566 at University of Southern California**

-----

## 📌 Project Overview

This project presents an end-to-end deep learning framework for classifying fine-grained mating behaviors in wild Ruffs (*Calidris pugnax*) using tri-axial accelerometer data.

Monitoring rare mating strategies (Independent, Satellite, Faeder) is critical for evolutionary biology but manual observation is unscalable. Biologging offers a solution, but standard ML models fail to generalize to unseen individuals due to **sensor misalignment** and **extreme class imbalance** (\<0.5% mating events).

We engineered a **DeepConvLSTM+** pipeline incorporating **Physics-Based Data Augmentation**, **Class-Balanced Focal Loss**, and **Test-Time Augmentation (TTA)**. Evaluating on a rigorous **Leave-One-Individual-Out (LSIO)** protocol, our approach achieves a **Macro F1 of 0.46**, surpassing the state-of-the-art benchmark (0.40) by 15%.

-----

## 🚀 Key Features

1.  **Physics-Based 3D Rotation Augmentation**

      * Simulates tag slippage during training by applying random 3D rotation matrices to raw signals.
      * Forces the model to learn *motion patterns* rather than specific sensor orientations.

2.  **Class-Balanced Focal Loss**

      * Replaces synthetic oversampling (which hurts precision) with a dynamic loss function.
      * Up-weights hard, rare classes (e.g., *Copulation*) and down-weights easy background classes (*Resting*).

3.  **Test-Time Augmentation (TTA) & Smoothing**

      * **Inference:** Predicts on 3 versions of the input (Original, Noisy, Shifted) and averages probabilities.
      * **Post-Processing:** Applies a Median Filter ($k=5$) to prediction streams to eliminate unrealistic label jitter.

4.  **Optimized Data Pipeline**

      * Handles **81GB** of raw data via efficient **SQL Batch Querying**.
      * Uses **90% Overlap** during windowing to maximize training data density (5x expansion).

-----

## 📊 Dataset

  * **Source:** 81GB Continuous 50Hz Tri-axial Accelerometer Data ($x, y, z$).
  * **Subjects:** 15 Captive Male Ruffs.
  * **Classes:** 13 Behaviors (e.g., Aggressive Posturing, Flying, Copulation Attempt, Resting).
  * **Protocol:** Leave-One-Individual-Out (LSIO). We train on 14 birds and test on 1 unseen bird, rotating through all 15.

-----

## 🛠️ Installation & Requirements

The project requires **Python 3.7+** and a GPU-enabled environment (e.g., Kaggle, Colab, or local CUDA setup).

### **Dependencies**

```bash
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib seaborn scikit-learn
pip install scipy tqdm prettytable
```

-----

## 💻 Usage

The entire pipeline is contained within a single, modularized script for ease of execution on Kaggle/Colab.

### **1. Configuration**

Modify the `Parameters` class to point to your data:

```python
class Parameters:
    def __init__(self):
        self.db_path = "/path/to/ruff-acc.db"
        self.labels_path = "/path/to/labels.csv"
        self.out_dir = "./output/"
        # ... hyperparams ...
```

### **2. Running the Pipeline**

Run the main execution block. The script automatically:

1.  Ingests data via SQL.
2.  Performs Sliding Window Segmentation.
3.  Runs the 15-Fold LSIO Cross-Validation.
4.  Generates Loss Curves and Confusion Matrices for every fold.

<!-- end list -->

```python
python main.py
```

-----

## 🧠 Model Architecture: DeepConvLSTM+

We enhanced the standard DeepConvLSTM architecture for better feature extraction and stability:

  * **Input:** $(Batch, 1, 50, 3)$ — 1 second of 3-axis data.
  * **Feature Extractor:** 2x Conv1D Layers (64 filters) + **Batch Normalization** + ReLU.
  * **Temporal Modeling:** 2x Stacked **LSTM** Layers (256 units) with Dropout (0.5).
  * **Classifier:** Fully Connected Layer $\to$ Softmax (13 Classes).

-----

## 📈 Results

| Metric | Baseline Paper (Aulsebrook et al.) | Our Model (DeepConvLSTM+) | Improvement |
| :--- | :--- | :--- | :--- |
| **Macro F1** | \~0.40 | **0.46** | **+15%** |
| **Accuracy** | \~53% | **58%** | **+5%** |
| **Rare Recall** | Negligible | **0.34 - 0.43** | **Significant** |

  * **Peak Performance:** Fold 8 achieved an **F1 of 0.52**.
  * **Rare Events:** Successfully recovered **43% of "Being Mounted"** events, providing the first reliable automated tool for monitoring rare mating strategies.

-----

## 🔮 Future Work

1.  **Transformer Architectures:** Replace LSTM with Self-Attention to better capture long-range temporal dependencies in complex mating dances.
2.  **Self-Supervised Learning:** Pre-train on the unlabeled portion of the 81GB dataset (Masked Autoencoders) to learn general motion representations.
3.  **Edge Deployment:** Quantize the model to run on-device (smart tags) for real-time monitoring.

-----

## 📚 References

1.  Aulsebrook, A. E., et al. (2024). *Quantifying mating behaviour using accelerometry and machine learning: challenges and opportunities.* Animal Behaviour.
2.  Riaboff, L., et al. (2019). *Predicting locomotion traits of dairy cows from accelerometer data.* Computers and Electronics in Agriculture.
3.  Vesalainen, E., et al. (2019). *Using Machine Learning for Remote Behaviour Classification in Cheetahs.* Ecology and Evolution.
