import os
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

def load_zinc_data(batch_size=32, data_root='data/ZINC'):
    """
    Load the ZINC dataset with PyTorch Geometric's DataLoader.
    Checks if the dataset is already downloaded and skips downloading if found.
    Args:
        batch_size (int): Batch size for the DataLoader.
        data_root (str): Path to the root directory for the dataset.
    Returns:
        train_loader, val_loader, test_loader: Data loaders for training, validation, and testing.
    """
    # Check if the dataset already exists
    if not os.path.exists(data_root):
        print(f"Dataset not found in {data_root}. Downloading...")
        dataset = ZINC(root=data_root)
    else:
        print(f"Dataset found in {data_root}. Skipping download.")
        dataset = ZINC(root=data_root)

    # Ensure node features and edge attributes are floats globally
    for data in dataset:
        data.x = data.x.float()  # Convert node features to float
        if data.edge_attr is not None:  # Check if edge attributes exist
            data.edge_attr = data.edge_attr.float()  # Convert edge attributes to float


    train_dataset = dataset[:15000]
    val_dataset = dataset[15000:16000]
    test_dataset = dataset[16000:]


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
