import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data_loader import extract_tumor_slices, ROIDataset
from src.models import Autoencoder
from src.utils import plot_T2_slices

# --- Configuration ---
ROOT_DIR = "./data"
METADATA_PATH = os.path.join(ROOT_DIR, "UCSF-PDGM-metadata_v5.csv")
LATENT_DIM = 28
BATCH_SIZE = 16
N_EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STATUS_TO_INT = {
    "relative co-deletion": 1,
    "co-deletion": 1,
    "intact": 2
}

def main():
    # prepare metadata
    metadata_df = pd.read_csv(METADATA_PATH)
    grade_2_3_df = metadata_df[metadata_df["WHO CNS Grade"].isin([2, 3])]
    ids_grade_2_3 = [id_str.split('-')[-1] for id_str in grade_2_3_df["ID"].tolist()]
    
    sample_to_status = dict(zip(ids_grade_2_3, grade_2_3_df["1p/19q"].tolist()))

    # data loading
    print("Extracting slices ...")
    roi_slices, labels = extract_tumor_slices(ROOT_DIR, ids_grade_2_3, sample_to_status, STATUS_TO_INT)
    
    # model
    dataset = ROIDataset(roi_slices, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    autoencoder = Autoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    
    # train
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    print(f"Starting training on {DEVICE}...")
    for epoch in range(N_EPOCHS):
        autoencoder.train()
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = autoencoder(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{N_EPOCHS} - Loss: {total_loss/len(dataloader):.6f}")

    # save model
    model_dir = "models" 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "autoencoder_glioma.pth")
    torch.save(autoencoder.state_dict(), model_path)
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()