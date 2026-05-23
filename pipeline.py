import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import joblib

# Target image size for MobileNetV2
IMG_SIZE = 224

# Setup standard Image Normalization for Pre-trained torchvision backbones (ImageNet stats)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_and_clean_tabular_data(info_txt_path: str) -> pd.DataFrame:
    """
    Parses and cleans the space-separated HousesInfo.txt tabular metadata.
    Columns: bedrooms, bathrooms, area, zipcode, price
    """
    if not os.path.exists(info_txt_path):
        raise FileNotFoundError(f"HousesInfo file not found at: {info_txt_path}")
        
    # Read space-separated values
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(info_txt_path, sep=" ", header=None, names=cols)
    
    # 1. Clean zipcodes (convert to string categories)
    df["zipcode"] = df["zipcode"].astype(str)
    
    # 2. Parse prices and square footage to float
    df["area"] = df["area"].astype(float)
    df["price"] = df["price"].astype(float)
    df["bedrooms"] = df["bedrooms"].astype(float)
    df["bathrooms"] = df["bathrooms"].astype(float)
    
    return df

class HousingDataset(Dataset):
    """
    Custom Multimodal PyTorch Dataset loading both house images and scaled tabular attributes.
    Returns: (image_tensor, tabular_tensor), target_price
    """
    def __init__(self, df: pd.DataFrame, image_dir: str, scaler: StandardScaler = None, zip_list: list = None, is_training: bool = True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.is_training = is_training
        
        # 1. Select numerical features to scale
        self.num_features = ["bedrooms", "bathrooms", "area"]
        
        # Fit or load tabular scaler (StandardScaler)
        if scaler is None and is_training:
            self.scaler = StandardScaler()
            self.scaled_num = self.scaler.fit_transform(self.df[self.num_features])
            # Save the scaler so app.py can load it for real-time inference
            joblib.dump(self.scaler, "tabular_scaler.joblib")
        elif scaler is not None:
            self.scaler = scaler
            self.scaled_num = self.scaler.transform(self.df[self.num_features])
        else:
            # Fallback if training but no scaler passed
            self.scaler = StandardScaler()
            self.scaled_num = self.scaler.fit_transform(self.df[self.num_features])

        # 2. Handle Zipcode categorical encoding using a shared vocabulary list
        # We one-hot encode zipcode for neural net ingestion. 
        if zip_list is None:
            self.zip_list = sorted(list(self.df["zipcode"].unique()))
            if is_training:
                # Save the zipcode vocabulary so app.py can load it dynamically for serving
                joblib.dump(self.zip_list, "zip_list.joblib")
        else:
            self.zip_list = zip_list
            
        # One-hot encode using reindex to match the shared zip_list columns exactly
        self.onehot_zips = pd.get_dummies(self.df["zipcode"]).reindex(columns=self.zip_list, fill_value=0).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Retrieve target index (1-indexed matching Eman Ahmed file system: e.g. "1_frontal.jpg")
        # In our DataFrame, the original row index is tracked or we can use the 1-indexed row number.
        # To be bulletproof, we will look at the index in the original dataframe + 1
        house_id = self.df.iloc[idx].name + 1
        image_name = f"{house_id}_frontal.jpg"
        image_path = os.path.join(self.image_dir, image_name)
        
        # 2. Robust Image Loading with fallback if frontal is missing
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            # Create a silent fallback solid color image if file is missing (safeguards against corrupted datasets)
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(240, 240, 240))
            
        image_tensor = IMAGE_TRANSFORM(image)
        
        # 3. Concatenate scaled numerical features and one-hot encoded zipcode features
        num_vec = self.scaled_num[idx].astype(np.float32)
        zip_vec = self.onehot_zips[idx]
        tabular_vec = np.concatenate([num_vec, zip_vec])
        tabular_tensor = torch.tensor(tabular_vec, dtype=torch.float32)
        
        # 4. Get target price
        price = self.df.iloc[idx]["price"]
        price_tensor = torch.tensor(price, dtype=torch.float32)
        
        return (image_tensor, tabular_tensor), price_tensor
