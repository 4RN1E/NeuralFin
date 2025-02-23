import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SharkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all classes and sort them for consistent indexing
        self.classes = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        
        # Create class to index mapping starting from 0
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Initialize lists to store paths and labels
        self.images = []
        self.labels = []
        
        # Load dataset
        print(f"\nLoading dataset from: {root_dir}")
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get valid images
            class_images = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            # Add images and labels
            for img_name in class_images:
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(class_idx)
        
        # Print dataset info
        print(f"Found {len(self.classes)} classes: {self.classes}")
        print(f"Class mapping: {self.class_to_idx}")
        print(f"Total images: {len(self.images)}")
        print(f"Label range: {min(self.labels)} to {max(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

if __name__ == '__main__':
    # Test the dataset
    dataset = SharkDataset('data/processed/train')
    print(f"Dataset size: {len(dataset)}")
    print(f"First item: {dataset[0]}")