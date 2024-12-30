# Graph Neural Network for Molecular Property Prediction

## Project Overview

This project implements a Graph Neural Network (GNN) using PyTorch Geometric to predict molecular properties from the ZINC dataset. The ZINC dataset is a benchmark for molecular property prediction tasks, containing molecular graph data with node and edge features. This work focuses on leveraging GNNs to capture the structural dependencies of molecular graphs for accurate property prediction.

## Objectives

- Design and train a GNN architecture suitable for molecular property prediction.
- Utilize the ZINC dataset for training, validation, and testing.
- Evaluate model performance based on Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## Methodology

### Project Structure

The project is modularly organized for scalability and clarity:

project/ 
├── models/ 
 └── gnn_model.py # Defines the GNN architecture. 
├── data/ 
   └── load_data.py # Utility script to load and split the ZINC dataset. 
├── utils/ 
    └── metrics.py # Evaluation metrics, including Mean Absolute Error. 
├── main.py # Main script to train and evaluate the model.
├── README.md # Project documentation and instructions. 
├── requirements.txt # Dependencies for the project.


### GNN Architecture

The GNN model is implemented in `models/gnn_model.py` with the following components:
- **Graph Convolution Layers**: Two `GCNConv` layers for extracting features from node and edge information.
- **Global Mean Pooling**: Aggregates node-level features into a single graph-level representation.
- **Linear Layer**: Maps the graph-level representation to the predicted property value.

### Dataset Handling

The ZINC dataset is loaded using PyTorch Geometric's `ZINC` class:
- **Training Data**: 15,000 molecular graphs.
- **Validation Data**: 1,000 molecular graphs.
- **Test Data**: 1,000 molecular graphs.

The dataset's node and edge features are preprocessed to ensure compatibility with PyTorch tensors.

### Training and Evaluation

- **Optimizer**: Adam optimizer with a learning rate of 0.0005.
- **Loss Function**: Mean Squared Error (MSE).
- **Metrics**: Mean Absolute Error (MAE) for model evaluation.

Early stopping is employed based on validation loss, with patience set to 10 epochs.

## Results

The model's performance is summarized below:

- **Training Loss (Final Epoch)**: 2.9459
- **Validation Loss (Final Epoch)**: 2.9041
- **Test Loss**: 2.8657
- **Test MAE**: 1.1475

The results demonstrate that the GNN effectively captures molecular graph structures and achieves competitive accuracy in property prediction.

## Challenges and Observations

- Initial data preprocessing issues were resolved by converting all node and edge features to `float` data type.
- Warnings regarding missing dependencies (e.g., `torch-scatter`, `torch-sparse`) did not significantly affect the results.
- Early stopping successfully prevented overfitting, enabling the model to generalize better on unseen data.

## Requirements

To replicate this project, install the required dependencies listed in `requirements.txt`:

torch torch-geometric torch-scatter torch-sparse

bash
Copier le code

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd project

Install dependencies:


pip install -r requirements.txt

Run the training script:
python main.py

# References


Resources and References

Graph Transformer Paper: https://proceedings.neurips.cc/paper_files/paper/2019/file/9d63484abb477c97640154d40595a3bb-Paper.pdf
ZINC Dataset Documentation: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ZINC.html
PyTorch Geometric GitHub: https://github.com/pyg-team/pytorch_geometric
Tutorial on GNNs: https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial
Review on Graph Neural Networks: https://arxiv.org/pdf/1812.08434.pdf

