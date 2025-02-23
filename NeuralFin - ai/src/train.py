import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datetime import datetime

from model import SharkNet
from dataset import SharkDataset

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}] | '
                  f'Loss: {loss.item():.4f} | '
                  f'Acc: {100. * correct / total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    print("\n🦈 SharkSense AI - Marine Habitat Protection System")
    
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = SharkDataset(root_dir='data/processed/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create model directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize model
    num_classes = len(train_dataset.classes)
    model = SharkNet(num_classes=num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\nStarting training...")
    start_time = datetime.now()
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        
        # Train one epoch
        epoch_loss, epoch_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"Epoch Summary:")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Accuracy: {epoch_acc:.2f}%")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'models/shark_model_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Training complete
    training_time = datetime.now() - start_time
    print(f"\nTraining completed! Total time: {training_time}")
    
    # Save final model
    final_path = 'models/shark_model_final.pth'
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")

if __name__ == "__main__":
    main()