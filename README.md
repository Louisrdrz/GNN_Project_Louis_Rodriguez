# Graph Neural Network for Molecular Property Prediction

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Methodology](#methodology)
   - [Project Structure](#project-structure)
   - [GNN Architecture](#gnn-architecture)
   - [Dataset Handling](#dataset-handling)
   - [Training and Evaluation](#training-and-evaluation)
4. [Results](#results)
5. [Challenges and Observations](#challenges-and-observations)
6. [Requirements](#requirements)
7. [How to Run](#how-to-run)
8. [References](#references)

---

## Project Overview

This project implements a Graph Neural Network (GNN) using PyTorch Geometric to predict molecular properties based on the ZINC dataset. The ZINC dataset serves as a benchmark for molecular property prediction tasks, containing molecular graph data with node and edge features. This project aims to leverage GNNs to capture the structural dependencies of molecular graphs for accurate property prediction.

---

## Objectives

- Design and train a GNN architecture tailored for molecular property prediction.
- Utilize the ZINC dataset for training, validation, and testing.
- Evaluate model performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

---

## Methodology

### Project Structure

The project follows a modular structure to ensure scalability and clarity:

```
project/
├── models/
│   └── gnn_model.py          # Defines the GNN architecture.
├── data/
│   └── load_data.py          # Utility script to load and split the ZINC dataset.
├── utils/
│   └── metrics.py            # Evaluation metrics, including Mean Absolute Error.
├── main.py                   # Main script to train and evaluate the model.
├── README.md                 # Project documentation and instructions.
├── requirements.txt          # Dependencies for the project.
```

### GNN Architecture

The GNN model is implemented in `models/gnn_model.py` and consists of the following components:

- **Graph Convolution Layers**: Two `GCNConv` layers to extract features from node and edge information.
- **Global Mean Pooling**: Aggregates node-level features into a single graph-level representation.
- **Linear Layer**: Maps the graph-level representation to the predicted property value.

### Dataset Handling

The ZINC dataset is loaded using PyTorch Geometric's `ZINC` class and split as follows:
- **Training Data**: 15,000 molecular graphs.
- **Validation Data**: 1,000 molecular graphs.
- **Test Data**: 1,000 molecular graphs.

Node and edge features are preprocessed to ensure compatibility with PyTorch tensors.

### Training and Evaluation

- **Optimizer**: Adam optimizer with a learning rate of 0.0005.
- **Loss Function**: Mean Squared Error (MSE).
- **Evaluation Metric**: Mean Absolute Error (MAE).

Early stopping is employed based on validation loss, with patience set to 10 epochs.

---

## Results

The model's performance is summarized as follows:

- **Training Loss (Final Epoch)**: 2.9459
- **Validation Loss (Final Epoch)**: 2.9041
- **Test Loss**: 2.8657
- **Test MAE**: 1.1475

These results demonstrate that the GNN effectively captures molecular graph structures and achieves competitive accuracy in property prediction.

---

## Challenges and Observations

- **Data Preprocessing**: Initial issues were resolved by converting all node and edge features to the `float` data type.
- **Dependencies**: Warnings regarding missing dependencies (e.g., `torch-scatter`, `torch-sparse`) did not significantly affect results.
- **Overfitting Prevention**: Early stopping successfully prevented overfitting, enabling better generalization to unseen data.

---

## Requirements

To replicate this project, install the required dependencies listed in `requirements.txt`:

```plaintext
torch
torch-geometric
torch-scatter
torch-sparse
```

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training script**:
   ```bash
   python main.py
   ```

---

## References

- [Graph Transformer Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/9d63484abb477c97640154d40595a3bb-Paper.pdf)
- [ZINC Dataset Documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ZINC.html)
- [PyTorch Geometric GitHub](https://github.com/pyg-team/pytorch_geometric)
- [Tutorial on GNNs](https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial)
- [Review on Graph Neural Networks](https://arxiv.org/pdf/1812.08434.pdf)

---
