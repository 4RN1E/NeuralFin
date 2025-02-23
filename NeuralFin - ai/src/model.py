import torch
import torch.nn as nn
import torchvision.models as models

class SharkNet(nn.Module):
    def __init__(self, num_classes):  # Remove default value to catch errors
        super().__init__()
        print(f"Initializing SharkNet with {num_classes} classes")
        
        # Initialize ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Modify final layer for classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        print(f"Model output size confirmed: {num_classes}")
    
    def forward(self, x):
        return self.model(x)