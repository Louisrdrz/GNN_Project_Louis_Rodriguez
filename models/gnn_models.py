import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNModel(torch.nn.Module):
    """
    GNN model with two GCN layers followed by global pooling and linear layers for property prediction.
    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        output_dim (int): Output dimension, typically 1 for regression tasks.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        return x
