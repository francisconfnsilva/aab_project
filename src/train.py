import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def train_autoencoder(model, dataloader, n_epochs=20, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            
            x_recon, _ = model(x)
            loss = criterion(x_recon, x)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            
        avg_loss = total_loss / len(dataloader.dataset)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
        
    return train_losses

def get_latent_features(model, roi_slices, device='cpu'):
    model.eval()
    features = []
    with torch.no_grad():
        for s in roi_slices:
            # Add batch and channel dims: (1, 1, H, W)
            x = torch.tensor(s).unsqueeze(0).unsqueeze(0).to(device)
            _, z = model(x)
            features.append(z.cpu().numpy().flatten())
    return np.array(features)