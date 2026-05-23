import torch
import torch.nn as nn
from torchvision import models

class MultimodalHousingPriceModel(nn.Module):
    """
    Late-Fusion Neural Network for Multimodal Housing Price Prediction.
    Branch 1: MobileNetV2 (pre-trained, frozen CNN layers) projecting visual assets to 64-dim embeddings.
    Branch 2: MLP network projecting tabular customer service features to 64-dim embeddings.
    Fusion Head: Concatenates both branches (128 dims) and regresses the estimated house price.
    """
    def __init__(self, tabular_input_dim: int):
        super(MultimodalHousingPriceModel, self).__init__()
        
        # 1. Visual Feature Extractor (MobileNetV2)
        # MobileNetV2 is extremely lightweight and accurate (~8MB weights)
        self.cnn = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Freeze baseline pre-trained weights to speed up training and prevent overfitting
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        # Reconstruct the classification head to project features to 64 dimensions
        self.cnn.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 2. Tabular Feature Extractor (MLP)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 3. Multimodal Late Fusion Regressor Head
        # Visual (64) + Tabular (64) = 128 dimensions input
        self.fusion_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Outputs a single floating point value: price
        )

    def forward(self, image_tensor, tabular_tensor):
        # 1. Forward pass through CNN branch
        # BatchNorm1d requires 2D tensors. In case of 1D evaluation batch size = 1,
        # PyTorch eval() takes care of tracking running averages.
        visual_emb = self.cnn(image_tensor)
        
        # 2. Forward pass through Tabular MLP branch
        tabular_emb = self.tabular_mlp(tabular_tensor)
        
        # 3. Concatenate along channel dimension
        fused_features = torch.cat((visual_emb, tabular_emb), dim=-1)
        
        # 4. Predict house price
        predicted_price = self.fusion_head(fused_features)
        
        # Squeeze output to (Batch,) dimension
        return predicted_price.squeeze(-1)
