# 24Fall-ML-Final: Federated Learning Exploration

This project explores the implementation of a **Federated Learning** framework for binary classification. It focuses on privacy-preserving model training across distributed datasets.

---

## Features

- Privacy-preserving preprocessing of categorical and numerical data.
- Simulation of federated learning with local and global model updates.
- Wide and Deep model combining linear and non-linear learning.
- Handles class imbalance with weighted loss computation.

---

## Dataset

The dataset includes:
- **Categorical Features**: `C1` to `C27`.
- **Numerical Features**: `I1` to `I14`.
- **Label**: Binary classification target.

Preprocessed files:
- `new_0.9_train.csv` (Training Data)
- `new_0.1_test.csv` (Test Data)

---

## Model Architecture

- **Wide Component**:
  - Linear layer for memorization of feature interactions.
- **Deep Component**:
  - Embedding layers for categorical features.
  - Fully connected layers for dense features.
- **Combined Output**:
  - Combines wide and deep outputs for the final prediction.

---

## Evaluation Metrics

- **Binary Cross-Entropy Loss**: Measures the performance of predictions.
- **Area Under the Curve (AUC)**: Evaluates classification performance.

---

## Files

- **Code**:
  - Preprocessing, model training, and evaluation scripts.
- **Data**:
  - Preprocessed training and testing datasets.
- **Logs**:
  - TensorBoard logs for tracking metrics.

---

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
