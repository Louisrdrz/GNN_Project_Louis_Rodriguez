import torch
from torch.optim import Adam
from models.gnn_models import GNNModel
from data.load_data import load_zinc_data
from utils.metrics import mean_absolute_error
import os

def train(model, loader, optimizer, criterion, device):
    """
    Train the GNN model for one epoch.
    """
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        data.x = data.x.float()  # Ensure node features are float
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr.float()
        
        optimizer.zero_grad()
        out = model(data)
        # Fix the size mismatch by squeezing the model output
        loss = criterion(out.squeeze(-1), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    """
    Evaluate the GNN model on validation or test data.
    """
    model.eval()
    total_loss = 0
    total_mae = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.x = data.x.float()
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr.float()
            
            out = model(data)
            loss = criterion(out.squeeze(-1), data.y)
            mae = mean_absolute_error(out.squeeze(-1), data.y)
            total_loss += loss.item()
            total_mae += mae
    return total_loss / len(loader), total_mae / len(loader)

def main():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    input_dim = 1  # Matches the ZINC dataset's node feature size
    hidden_dim = 128  # Increased hidden dimensions for better capacity
    output_dim = 1  # Single target value for regression
    epochs = 100
    batch_size = 64  # Larger batch size for stable gradients
    learning_rate = 0.0005

    # Load data
    train_loader, val_loader, test_loader = load_zinc_data(batch_size=batch_size, data_root='data/ZINC')

    # Initialize model, criterion, and optimizer
    model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Early stopping setup
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load the best model and test
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_mae = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

if __name__ == "__main__":
    main()
