import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import math

# Dynamically inject the parent directory into sys.path to guarantee local module resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import load_and_clean_tabular_data, HousingDataset
from model import MultimodalHousingPriceModel

def ensure_dataset():
    """
    Checks if the Eman Ahmed Houses Dataset is available locally.
    If missing, programmatically clones the repository to ensure seamless execution.
    """
    if not os.path.exists("Houses-dataset"):
        print("Cloning Houses-dataset repository from GitHub...")
        os.system("git clone https://github.com/emanhamed/Houses-dataset.git")
    else:
        print("✔ Dataset directory found.")

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    total_samples = 0
    
    for (images, tabular), targets in dataloader:
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(images, tabular)
        loss = loss_fn(predictions, targets)
        
        # Backward pass & Optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * targets.size(0)
        running_mae += torch.abs(predictions - targets).sum().item()
        total_samples += targets.size(0)
        
    epoch_mse = running_loss / total_samples
    epoch_mae = running_mae / total_samples
    epoch_rmse = math.sqrt(epoch_mse)
    return epoch_mae, epoch_rmse

def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for (images, tabular), targets in dataloader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)
            
            predictions = model(images, tabular)
            loss = loss_fn(predictions, targets)
            
            running_loss += loss.item() * targets.size(0)
            running_mae += torch.abs(predictions - targets).sum().item()
            total_samples += targets.size(0)
            
    epoch_mse = running_loss / total_samples
    epoch_mae = running_mae / total_samples
    epoch_rmse = math.sqrt(epoch_mse)
    return epoch_mae, epoch_rmse

def main():
    # 1. Guarantee dataset exists
    ensure_dataset()
    
    # Paths according to Eman Ahmed repository layout
    info_path = os.path.join("Houses-dataset", "Houses Dataset", "HousesInfo.txt")
    image_dir = os.path.join("Houses-dataset", "Houses Dataset")
    
    # 2. Load and preprocess tabular features
    df = load_and_clean_tabular_data(info_path)
    
    # Split dataset into train (80%) and validation (20%)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training instances  : {len(train_df)}")
    print(f"Validation instances: {len(val_df)}")
    
    # 3. Build PyTorch Dataset loaders
    train_dataset = HousingDataset(train_df, image_dir, is_training=True)
    # Share training scaler and zipcode vocabulary with validation set to prevent data leakage and shape mismatches!
    val_dataset = HousingDataset(val_df, image_dir, scaler=train_dataset.scaler, zip_list=train_dataset.zip_list, is_training=False)
    
    # Set drop_last=True for Dataloader to avoid Batch Size = 1 issues with BatchNorm1d during training
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 4. Instantiate model
    # Tabular input dimension = 3 scaled numeric features + number of zipcodes (from one-hot columns)
    tabular_dim = 3 + len(train_dataset.zip_list)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Multimodal Network on: {device.type.upper()}")
    
    model = MultimodalHousingPriceModel(tabular_input_dim=tabular_dim).to(device)
    
    # 5. Setup Optimizers
    loss_fn = nn.MSELoss()
    # High learning rate for the newly initialized head, CNN backbone is frozen
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # 6. Training loop
    EPOCHS = 15
    best_val_mae = float("inf")
    
    print("\n--- Starting Multimodal Neural Net Training ---")
    for epoch in range(EPOCHS):
        t_mae, t_rmse = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        v_mae, v_rmse = evaluate_model(model, val_loader, loss_fn, device)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | Train MAE: ${t_mae:,.2f} | Train RMSE: ${t_rmse:,.2f} | Val MAE: ${v_mae:,.2f} | Val RMSE: ${v_rmse:,.2f}")
        
        # Save best model checkpoint based on validation MAE
        if v_mae < best_val_mae:
            best_val_mae = v_mae
            print(f" --> Saving new best model checkpoint...")
            torch.save(model.state_dict(), "multimodal_house_model.pth")
            
    print(f"\n✔ Training Completed! Best Validation MAE: ${best_val_mae:,.2f}")
    print("Model weights saved successfully to: 'multimodal_house_model.pth'")

if __name__ == "__main__":
    main()
